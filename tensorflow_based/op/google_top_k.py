import tensorflow as tf
import os, sys
import numpy as np
sys.path.append("./op")
from op.isotonic_dykstra import isotonic_dykstra_mask


def sparse_soft_topk_mask_dykstra(x, k, l=1e-1, num_iter=500):
    """TensorFlow implementation of differentiable Top-K mask operation (Dykstra algorithm), supports 1D and 2D input"""
    original_shape = x.shape
    if len(original_shape) == 1:
        x = tf.expand_dims(x, 0)  # Add batch dimension
        
    batch_size = tf.shape(x)[0]
    n = tf.shape(x)[1]
    
    # Get sorted indices and permutation matrix
    perm = tf.argsort(-x, axis=1)
    P = tf.one_hot(perm, n, dtype=tf.float32)
    
    # Compute sorted values
    s = tf.linalg.matmul(P, tf.expand_dims(x, -1))
    s = tf.squeeze(s, -1)
    
    # Create weight matrix
    w = tf.concat([
        tf.ones((batch_size, k), dtype=x.dtype),
        tf.zeros((batch_size, n - k), dtype=x.dtype)
    ], axis=1)
    
    s_w = s - l * w
    
    # Apply isotonic regression
    out_dykstra = isotonic_dykstra_mask(s_w, num_iter)
    out = (s - out_dykstra) / l
    
    # Reconstruct the output
    result = tf.linalg.matmul(tf.transpose(P, [0, 2, 1]), tf.expand_dims(out, -1))
    result = tf.squeeze(result, -1)
    
    # Restore original shape if input was 1D
    if len(original_shape) == 1:
        return tf.squeeze(result, 0)
    return result

def sparse_soft_topk_mag_dykstra(x, k, l=1e-1, num_iter=500):
    """TensorFlow implementation of differentiable magnitude Top-K operation (Dykstra algorithm), supports 1D and 2D input"""
    original_shape = x.shape
    if len(original_shape) == 1:
        x = tf.expand_dims(x, 0)
        
    batch_size = tf.shape(x)[0]
    n = tf.shape(x)[1]
    
    # Get sorted indices of absolute values
    perm = tf.argsort(-tf.abs(x), axis=1)
    P = tf.one_hot(perm, n, dtype=tf.float32)
    
    # Compute sorted absolute values
    abs_x = tf.abs(x)
    s = tf.linalg.matmul(P, tf.expand_dims(abs_x, -1))
    s = tf.squeeze(s, -1)
    
    # Create weight matrix and adjust sorted values
    w = tf.pad(tf.ones((batch_size, k)), [[0, 0], [0, n - k]])
    adjusted_s = s / (1 + l * w)
    
    # Apply isotonic regression
    out_dykstra = isotonic_dykstra_mask(adjusted_s, num_iter)
    out = (s - out_dykstra) / l
    
    # Reconstruct and apply sign
    perm_out = tf.linalg.matmul(tf.transpose(P, [0, 2, 1]), tf.expand_dims(out, -1))
    perm_out = tf.squeeze(perm_out, -1)
    result = tf.sign(x) * perm_out * (1 + l)
    
    # Restore original shape if input was 1D
    if len(original_shape) == 1:
        return tf.squeeze(result, 0)
    return result

def hard_topk_mask(x, k):
    """Hard Top-K mask, supports 1D and 2D input"""
    if len(x.shape) == 1:
        x = tf.expand_dims(x, 0)
        
    # Get top-k indices and create one-hot mask
    values, indices = tf.math.top_k(x, k=k)
    result = tf.reduce_sum(tf.one_hot(indices, tf.shape(x)[1], dtype=tf.float32), axis=1)
    
    # Restore original shape if input was 1D
    if len(x.shape) == 1:
        return tf.squeeze(result, 0)
    return result

def hard_topk_mag(x, k):
    """Hard magnitude Top-K, supports 1D and 2D input"""
    return x * hard_topk_mask(tf.abs(x), k)

# # Test code
if __name__ == "__main__":
    # 1D test
    values_1d = tf.constant([4., 3., 2., 1., 6.])
    
    print("1D soft top-k mask:", sparse_soft_topk_mask_dykstra(values_1d, k=2, l=1).numpy().round(8))
    
    # 2D test
    values_2d = tf.constant([[4., 3., 2., 1., 6.], [1., 2., 3., 4., 9.]])
    tf.print("2D soft top-k mask:\n", sparse_soft_topk_mask_dykstra(values_2d, k=2, l=1).numpy().round(8))