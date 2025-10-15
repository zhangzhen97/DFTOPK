import tensorflow as tf

EPS = tf.constant(1e-8, dtype=tf.float32)
NEG_INF = tf.constant(-1e9, dtype=tf.float32)

def DFTopK(x, k, tau=1.0):
    values, _ = tf.nn.top_k(x, k=k+1)
    x_k = values[:, k:k+1]
    x_k_plus_1 = values[:, k-1:k]
    threshold = (x_k + x_k_plus_1) / 2.0
    logits = x - threshold
    if tau != 1:
        logits = logits / tau
    prob = tf.sigmoid(logits)
    return prob