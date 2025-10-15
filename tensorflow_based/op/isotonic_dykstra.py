import tensorflow as tf

def _cumsum_triu(x):
    """TensorFlow implementation of triangular cumsum"""
    mask = tf.linalg.band_part(tf.ones(tf.shape(x), 0, -1))  # Upper triangular mask
    return tf.einsum('ij,jk->ik', x, mask)

def _jvp_isotonic_mask(solution, vector, eps=1e-4):
    """Jacobian-vector product for isotonic regression"""
    x = solution
    mask = tf.pad(tf.math.abs(tf.experimental.numpy.diff(x)) <= eps, [(1, 0)], constant_values=False)
    ar = tf.range(tf.shape(x)[0], dtype=tf.int32)
    
    inds_start = tf.where(tf.math.logical_not(mask), ar, tf.cast(tf.fill(tf.shape(ar), tf.int32.max), dtype=tf.int32))
    inds_start = tf.sort(inds_start)
    
    one_hot_start = tf.one_hot(inds_start, depth=tf.shape(vector)[0], dtype=tf.float32)
    a = _cumsum_triu(one_hot_start)
    a_diff = tf.experimental.numpy.diff(a[::-1], axis=0)[::-1]
    a = tf.concat([a_diff, tf.expand_dims(a[-1], axis=0)], axis=0)
    return tf.reduce_sum((tf.transpose(a) * (tf.linalg.matvec(a, vector))) / (tf.reduce_sum(a, axis=1, keepdims=True) + 1e-8), axis=0)

def isotonic_dykstra_mag(s, w, l=1e-1, num_iter=500):
    """Weighted isotonic regression in TensorFlow"""
    def f(v, u):
        d = v[1::2] - v[::2]
        s_num = (v * u)[::2] + (v * u)[1::2]
        s_den = u[::2] + u[1::2]
        
        mask = tf.repeat(d < 0, 2)
        mean = tf.repeat(s_num / s_den, 2)
        return v * mask + mean * tf.cast(tf.math.logical_not(mask), dtype=tf.float32)
    
    u = 1 + l * w
    
    def body_fn(vpq):
        xk, pk, qk = vpq
        yk = tf.pad(f(xk[:-1] + pk[:-1], u[:-1]), [(0, 1)])
        yk = tf.tensor_scatter_nd_update(yk, [[tf.shape(yk)[0]-1]], [xk[-1] + pk[-1]])
        p = xk + pk - yk
        
        v = tf.pad(f(yk[1:] + qk[1:], u[1:]), [(1, 0)])
        v = tf.tensor_scatter_nd_update(v, [[0]], [yk[0] + qk[0]])
        q = yk + qk - v
        return v, p, q
    
    # Ensure odd length
    n = tf.shape(s)[0]
    if n % 2 == 0:
        minv = tf.reduce_min(s) - 1
        s = tf.pad(s, [(0, 1)])
        s = tf.tensor_scatter_nd_update(s, [[n]], [minv])
        u = tf.pad(u, [(0, 1)])
    
    v = tf.identity(s)
    p = tf.zeros_like(s)
    q = tf.zeros_like(s)
    vpq = (v, p, q)
    
    for _ in range(num_iter // 2):
        vpq = body_fn(vpq)
    
    sol = vpq[0]
    return sol if n % 2 != 0 else sol[:-1]

def _jvp_isotonic_mag(solution, vector, w, l, eps=1e-4):
    """Jacobian-vector product for weighted isotonic regression"""
    x = solution
    mask = tf.pad(tf.math.abs(tf.experimental.numpy.diff(x)) <= eps, [(1, 0)], constant_values=False)
    ar = tf.range(tf.shape(x)[0], dtype=tf.int32)
    
    inds_start = tf.where(tf.math.logical_not(mask), ar, tf.cast(tf.fill(tf.shape(ar), tf.int32.max), dtype=tf.int32))
    inds_start = tf.sort(inds_start)
    
    u = 1 + l * w
    one_hot_start = tf.one_hot(inds_start, depth=tf.shape(vector)[0], dtype=tf.float32)
    a = _cumsum_triu(one_hot_start)
    a_diff = tf.experimental.numpy.diff(a[::-1], axis=0)[::-1]
    a = tf.concat([a_diff, tf.expand_dims(a[-1], axis=0)], axis=0)
    return tf.reduce_sum((tf.transpose(a) * (tf.linalg.matvec(a, vector * u))) / (tf.reduce_sum(a * u, axis=1, keepdims=True) + 1e-8), axis=0)

def isotonic_dykstra_mask(s, num_iter=500):
    """TensorFlow implementation of isotonic regression using Dykstra's projection algorithm.
    Supports 2D input (batch processing).
    
    Args:
        s: input tensor of shape (batch_size, n) or (n,)
        num_iter: number of iterations
        
    Returns:
        sol: solution tensor of same shape as s
    """
    if len(s.shape) == 1:
        s = tf.expand_dims(s, 0)  # Add batch dimension
        
    batch_size = tf.shape(s)[0]
    n = tf.shape(s)[1]
    
    def f(v):
        """保持原始计算逻辑，同时严格保证输出形状与输入一致"""
        # 1. 输入保障和形状锁定
        v = tf.convert_to_tensor(v, dtype=tf.float32)
        input_shape = tf.shape(v)
        batch_size, original_length = input_shape[0], input_shape[1]
        
        # 2. 核心计算逻辑（保持原有操作）
        def _slice_and_operate(v, op):
            v_odd = v[:, 1::2]  # 奇数列 [B, L//2]
            v_even = v[:, ::2]   # 偶数列 [B, (L+1)//2]
            min_len = tf.minimum(tf.shape(v_odd)[1], tf.shape(v_even)[1])
            return op(v_odd[:, :min_len], v_even[:, :min_len])
        
        d = _slice_and_operate(v, tf.subtract)
        a = _slice_and_operate(v, tf.add)
        
        # 3. 形状恢复的严格实现
        def _strict_restore(x, target_length):
            current_len = tf.shape(x)[1]
            # 计算需要重复的次数（向上取整）
            repeat_times = tf.maximum(1, (target_length + current_len - 1) // current_len)
            expanded = tf.repeat(x, repeats=repeat_times, axis=1)
            # 物理截断确保形状精确匹配
            return expanded[:, :target_length]
        
        mask = _strict_restore(d < 0, original_length)
        mean = _strict_restore(a / 2.0, original_length)
        
        # 4. 计算结果（三重形状保证）
        result = tf.where(
            mask,
            v,  # 直接使用输入v保证形状
            mean
        )
        
        # 5. 强制形状一致性（关键！）
        result = tf.ensure_shape(result, [batch_size, original_length])
        result = result[:, :original_length]  # 物理截断
        
        return result

    def body_fn(vpq):
        xk, pk, qk = vpq
        # Modified pad operation
        yk = tf.pad(f(xk[:, :-1] + pk[:, :-1]), [(0, 0), (0, 1)])
        last_vals = tf.expand_dims(xk[:, -1] + pk[:, -1], axis=-1)
        yk = tf.concat([yk[:, :-1], last_vals], axis=1)
        
        p = xk + pk - yk
        
        v = tf.pad(f(yk[:, 1:] + qk[:, 1:]), [(0, 0), (1, 0)])
        first_vals = tf.expand_dims(yk[:, 0] + qk[:, 0], axis=-1)
        v = tf.concat([first_vals, v[:, 1:]], axis=1)
        
        q = yk + qk - v
        return v, p, q
    
    # Ensure odd length
    if n % 2 == 0:
        minv = tf.reduce_min(s, axis=1, keepdims=True) - 1
        s = tf.pad(s, [(0, 0), (0, 1)])
        s = tf.concat([s[:, :-1], minv], axis=1)
    
    v = tf.identity(s)
    p = tf.zeros_like(s)
    q = tf.zeros_like(s)
    vpq = (v, p, q)
    
    for _ in range(num_iter // 2):
        vpq = body_fn(vpq)
    
    sol = vpq[0]
    if n % 2 != 0:
        result = sol
    else:
        result = sol[:, :-1]
    
    s = tf.cond(
        tf.equal(tf.rank(s), 1),
        lambda: tf.expand_dims(s, 0),
        lambda: s
    )
    return result