import tensorflow as tf

def fake_sort_with_grad(x):
    sorted_x = tf.sort(x, axis=-1)
    return tf.stop_gradient(sorted_x) + tf.ones_like(x) * (sorted_x - tf.stop_gradient(sorted_x))

def SuTopK(x, k, stop_lamb=False):
    shape = tf.shape(x)
    batch_size = shape[0]
    n = shape[1]

    sorted_vals= fake_sort_with_grad(x)
    x_sorted = tf.reverse(sorted_vals, axis=[1])
    def stable_logcumsumexp(x, axis, eps=1e-12):
        m = tf.stop_gradient(tf.reduce_max(x, axis=axis, keepdims=True))
        x_shifted = x - m
        exp_x = tf.exp(tf.clip_by_value(x_shifted, -50, 50))
        cumsum = tf.cumsum(exp_x + eps, axis=axis)
        return m + tf.log(cumsum)
    
    lse1 = stable_logcumsumexp(x_sorted, axis=1)
    lse2_temp = stable_logcumsumexp(-tf.reverse(x_sorted, axis=[1]), axis=1)
    lse2 = tf.reverse(lse2_temp, axis=[1])
    lse2 = tf.concat([
        lse2[:, 1:], 
        tf.fill((batch_size, 1), tf.constant(-float('inf'), dtype=x.dtype))
    ], axis=1)

    m = tf.cast(tf.range(n - 1, -1, -1), tf.float32)
    m = tf.reshape(m, (1, n))
    diff = tf.cast(k, tf.float32) - m

    lse_sum = lse1 + lse2
    sqrt_arg = tf.square(diff) + tf.exp(tf.clip_by_value(lse_sum, -50, 50))
    sqrt_term = tf.sqrt(tf.maximum(sqrt_arg, 1e-15))
    
    log_arg = sqrt_term + diff + 1e-15
    x_lamb = lse1 - tf.log(log_arg)

    x_sorted_shift = tf.concat([
        x_sorted[:, 1:],
        tf.fill((batch_size, 1), tf.constant(float('inf'), dtype=x.dtype))
    ], axis=1)
    
    mask = tf.logical_and(
        x_lamb >= x_sorted,
        x_lamb <= x_sorted_shift
    )

    # 收集有效索引
    index = tf.tile(tf.expand_dims(tf.range(n), 0), [batch_size, 1])
    masked_idx = tf.where(mask, index, -tf.ones_like(index))
    last_valid_idx = tf.reduce_max(masked_idx, axis=1)

    invalid_mask = tf.equal(last_valid_idx, -1)
    safe_idx = tf.where(invalid_mask, 
                       tf.zeros_like(last_valid_idx), 
                       last_valid_idx)
    
    batch_indices = tf.stack([
        tf.range(batch_size, dtype=tf.int32),
        tf.cast(safe_idx, tf.int32)
    ], axis=1)
    lamb = tf.gather_nd(x_lamb, batch_indices)
    lamb = tf.expand_dims(lamb, 1)

    default_lamb = tf.reduce_mean(x, axis=1, keepdims=True)
    lamb = tf.where(tf.expand_dims(invalid_mask, 1), default_lamb, lamb)
    
    if stop_lamb:
        lamb = tf.stop_gradient(lamb)

    diff = x - lamb
    out = (1 - tf.exp(-tf.abs(diff))) * tf.sign(diff) * 0.5 + 0.5
    return tf.clip_by_value(out, 0.0, 1.0)