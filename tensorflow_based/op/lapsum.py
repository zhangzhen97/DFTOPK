import tensorflow as tf

def stable_log_cumsum_exp(x, axis=1, eps=1e-12):
    """
    稳定的 log(cumsum(exp(x)))，不依赖 r 范围
    """
    max_x = tf.stop_gradient(tf.reduce_max(x, axis=axis, keepdims=True))
    x_shifted = x - max_x
    exp_x = tf.exp(tf.clip_by_value(x_shifted, -50.0, 50.0))
    cumsum = tf.cumsum(exp_x + eps, axis=axis)
    return tf.math.log(cumsum) + max_x

def lapsum_top_k(r, k, alpha=1e3, descending=False):
    """
    数值稳定 soft top-k
    Args:
        r: [B, N] 输入 logits
        k: 标量 top-k
        alpha: 温度系数
        descending: 是否取降序 top-k
    Returns:
        p_final: [B, N] soft top-k 概率
    """
    r = tf.convert_to_tensor(r, dtype=tf.float32)
    batch_size = tf.shape(r)[0]
    num_dim = tf.shape(r)[1]

    # 排序
    neg_r = -r
    topk = tf.nn.top_k(neg_r, k=num_dim, sorted=True)
    asc = -topk.values
    scaled = asc / alpha

    # eB
    eB_log = stable_log_cumsum_exp(scaled, axis=1)
    eB = tf.exp(eB_log - scaled)

    # eA
    neg_scaled = -scaled
    flipped = tf.reverse(neg_scaled, axis=[1])
    eA_log_flipped = stable_log_cumsum_exp(flipped, axis=1)
    eA_log = tf.reverse(eA_log_flipped, axis=[1])
    eA = tf.exp(eA_log + scaled)

    # row: [1,3,5,...]
    row = tf.cast(tf.range(1, 2 * num_dim + 1, 2), tf.float32)
    row = tf.reshape(row, [1, -1])
    x_for_search = eA - eB + row

    # w
    if descending:
        w = tf.cast(k, tf.float32)
    else:
        w = tf.cast(num_dim - k, tf.float32)
    w = tf.reshape(tf.tile([w], [batch_size]), [-1, 1])

    # searchsorted
    i = tf.reduce_sum(tf.cast(x_for_search < 2.0 * w, tf.int32), axis=1)  # [B]

    m = tf.clip_by_value(i - 1, 0, num_dim - 1)
    n = tf.clip_by_value(i, 0, num_dim - 1)
    m = tf.reshape(m, [-1, 1])
    n = tf.reshape(n, [-1, 1])

    # batch gather
    batch_idx = tf.reshape(tf.range(batch_size), [-1, 1])
    idx_m = tf.concat([batch_idx, m], axis=1)
    idx_n = tf.concat([batch_idx, n], axis=1)

    s_m = tf.gather_nd(scaled, idx_m)
    s_n = tf.gather_nd(scaled, idx_n)

    a_full = tf.gather_nd(eA, idx_n)
    b_full = tf.gather_nd(eB, idx_m)
    a = tf.where(i < num_dim, a_full, tf.zeros_like(a_full))
    b = tf.where(i > 0, b_full, tf.zeros_like(b_full))

    e_val = tf.reshape(w, [-1]) - tf.cast(i, tf.float32)

    # 核心求解
    def _solve_tf(s, t, a, b, e):
        z = tf.sqrt(tf.maximum(e * e + a * b * tf.exp(tf.clip_by_value(s - t, -50.0, 50.0)), 1e-12)) + tf.abs(e)
        ab = tf.where(e > 0, a, b)
        out = tf.where(
            e > 0,
            t + tf.math.log(z + 1e-12) - tf.math.log(ab + 1e-12),
            s - tf.math.log(z + 1e-12) + tf.math.log(ab + 1e-12),
        )
        return out

    b_res = _solve_tf(s_m, s_n, a, b, e_val)
    b_res = tf.reshape(b_res, [-1, 1])

    sign = -1.0 if descending else 1.0
    x = r / (alpha * sign) - sign * b_res
    sign_x = x > 0
    p_abs = tf.abs(x)
    p = tf.exp(-p_abs) * 0.5
    p_final = tf.where(sign_x, 1.0 - p, p)

    # clip 输出保证交叉熵稳定
    p_final = tf.clip_by_value(p_final, 1e-6, 1.0 - 1e-6)
    return p_final
