import tensorflow as tf
import math
import numpy as np
from typing import Union, Any, Dict, List

SORTING_NETWORK_TYPE = List[tf.Tensor]


class SortNet:
    def __init__(self,sort_op,reverse=False,config=None):
        """
        reverse=True 是从小到大排，=False是从大到小排
        """
        self.sort_op=sort_op
        self.config=config
        self.reverse=reverse
        if sort_op=='neural_sort':
            self.net=NeuralSortNet(tau=self.config['tau'],col_unimodal=False,descending=True)
        elif sort_op=='soft_sort':
            self.net=SoftSortNet(tau=self.config['tau'],col_unimodal=False,descending=True)
        elif sort_op == 'neural_sort_col':
            self.net=NeuralSortNet(tau=self.config['tau'],col_unimodal=True,descending=True)
        elif sort_op=='soft_sort_col':
            self.net=SoftSortNet(tau=self.config['tau'],col_unimodal=True,descending=True)
        elif sort_op=='odd_even':
            self.net=DiffSortNet(sorting_network_type='odd_even',size=config['size'],
                    device=config['device'],steepness=config['steepness'],
                    art_lambda=config['art_lambda'],interpolation_type=config['interpolation_type'],
                    distribution=config['distribution'])
        elif sort_op=='bitonic':
            self.net = DiffSortNet(sorting_network_type='bitonic', size=config['size'],
                                   device=config['device'], steepness=config['steepness'],
                                   art_lambda=config['art_lambda'], interpolation_type=config['interpolation_type'],
                                   distribution=config['distribution'])
        else:
            raise NotImplementedError('[ERROR] sort_op `{}` unknown'.format(sort_op))

    def forward(self, x):
        permutation_matrix = self.net.forward(x)
        if (self.reverse and self.sort_op in ['neural_sort','neural_sort_col','soft_sort','soft_sort_col']) or (not self.reverse and self.sort_op=='odd_even') or (not self.reverse and self.sort_op=='bitonic'):
            return tf.reverse_v2(permutation_matrix,axis=[-2])
        else:
            return permutation_matrix

    @staticmethod
    def get_default_config(sort_op):
        if sort_op=='neural_sort':
            return NeuralSortNet.get_default_config()
        elif sort_op in ['odd_even', 'bitonic']:
            return DiffSortNet.get_default_config()
        elif sort_op == 'soft_sort':
            return SoftSortNet.get_default_config()
        else:
            raise NotImplementedError('[ERROR] sort_op `{}` unknown'.format(sort_op))
        
def bitonic_network(n):
    IDENTITY_MAP_FACTOR = .5
    num_blocks = math.ceil(np.log2(n))
    assert n <= 2 ** num_blocks
    network = []

    for block_idx in range(num_blocks):
        for layer_idx in range(block_idx + 1):
            m = 2 ** (block_idx - layer_idx)

            split_a, split_b = np.zeros((n, 2**num_blocks)), np.zeros((n, 2**num_blocks))
            combine_min, combine_max = np.zeros((2**num_blocks, n)), np.zeros((2**num_blocks, n))
            count = 0

            for i in range(0, 2**num_blocks, 2*m):
                for j in range(m):
                    ix = i + j
                    a, b = ix, ix + m

                    # Cases to handle n \neq 2^k: The top wires are discarded and if a comparator considers them, the
                    # comparator is ignored.
                    if a >= 2**num_blocks-n and b >= 2**num_blocks-n:
                        split_a[count, a], split_b[count, b] = 1, 1
                        if (ix // 2**(block_idx + 1)) % 2 == 1:
                            a, b = b, a
                        combine_min[a, count], combine_max[b, count] = 1, 1
                        count += 1
                    elif a < 2**num_blocks-n and b < 2**num_blocks-n:
                        pass
                    elif a >= 2**num_blocks-n and b < 2**num_blocks-n:
                        split_a[count, a], split_b[count, a] = 1, 1
                        combine_min[a, count], combine_max[a, count] = IDENTITY_MAP_FACTOR, IDENTITY_MAP_FACTOR
                        count += 1
                    elif a < 2**num_blocks-n and b >= 2**num_blocks-n:
                        split_a[count, b], split_b[count, b] = 1, 1
                        combine_min[b, count], combine_max[b, count] = IDENTITY_MAP_FACTOR, IDENTITY_MAP_FACTOR
                        count += 1
                    else:
                        assert False

            split_a = split_a[:count, 2 ** num_blocks - n:]
            split_b = split_b[:count, 2 ** num_blocks - n:]
            combine_min = combine_min[2**num_blocks-n:, :count]
            combine_max = combine_max[2**num_blocks-n:, :count]
            network.append((split_a, split_b, combine_min, combine_max))

    return network


def odd_even_network(n):
    layers = n

    network = []

    shifted: bool = False
    even: bool = n % 2 == 0

    for layer in range(layers):

        if even:
            k = n // 2 + shifted
        else:
            k = n // 2 + 1

        split_a, split_b = np.zeros((k, n)), np.zeros((k, n))
        combine_min, combine_max = np.zeros((n, k)), np.zeros((n, k))

        count = 0

        # for i in range(n // 2 if not (even and shifted) else n // 2 - 1):
        for i in range(int(shifted), n-1, 2):
            a, b = i, i + 1
            split_a[count, a], split_b[count, b] = 1, 1
            combine_min[a, count], combine_max[b, count] = 1, 1
            count += 1

        if even and shifted:
            # Make sure that the corner values stay where they are/were:
            a, b = 0, 0
            split_a[count, a], split_b[count, b] = 1, 1
            combine_min[a, count], combine_max[b, count] = .5, .5
            count += 1
            a, b = n - 1, n - 1
            split_a[count, a], split_b[count, b] = 1, 1
            combine_min[a, count], combine_max[b, count] = .5, .5
            count += 1

        elif not even:
            if shifted:
                a, b = 0, 0
            else:
                a, b = n - 1, n - 1
            split_a[count, a], split_b[count, b] = 1, 1
            combine_min[a, count], combine_max[b, count] = .5, .5
            count += 1

        assert count == k

        network.append((split_a, split_b, combine_min, combine_max))
        shifted = not shifted

    return network


def get_sorting_network(type, n, device):
    def matrix_to_tensor(m):
        with tf.device(device):
            return [[tf.convert_to_tensor(matrix,dtype=tf.float32) for matrix in matrix_set] for matrix_set in m]

    if type == 'bitonic':
        return matrix_to_tensor(bitonic_network(n))
    elif type == 'odd_even':
        return matrix_to_tensor(odd_even_network(n))
    else:
        raise NotImplementedError('Sorting network `{}` unknown.'.format(type))


def s_best(x):
    return tf.clip_by_value(x, -0.25, 0.25) + .5 + \
           tf.cast(x > 0.25, tf.float32) - (tf.cast(x < -0.25, tf.float32)) * (0.25 - 1 / 16 / (tf.abs(x) + 1e-10))


class NormalCDF:
    def __init__(self, sigma):
        self.sigma = sigma
        self.op = self.__get_op()

    def __get_op(self):
        # 使用 tf.custom_gradient() 函数定义一个新的乘法运算，并使用上面定义的梯度函数
        @tf.custom_gradient
        def _NormalCDF(x):
            def forward(x):
                return 0.5 + 0.5 * tf.math.erf(x / self.sigma / tf.math.sqrt(2.0))

            def backward(grad):
                return grad * 1 / self.sigma / tf.math.sqrt(math.pi * 2) * tf.exp(-0.5 * tf.pow(x / self.sigma, 2.0))

            return forward(x), backward

        return _NormalCDF

    def call(self, x):
        return self.op(x)


def execute_sort(
        sorting_network,
        vectors,
        steepness=10.,
        art_lambda=0.25,
        distribution='cauchy'
):
    x = vectors
    X = tf.eye(tf.shape(vectors)[1], dtype=x.dtype, batch_shape=[tf.shape(vectors)[0]])

    for split_a, split_b, combine_min, combine_max in sorting_network:
        split_a = tf.cast(split_a, x.dtype)
        split_b = tf.cast(split_b, x.dtype)
        combine_min = tf.cast(combine_min, x.dtype)
        combine_max = tf.cast(combine_max, x.dtype)

        a, b = tf.matmul(x, split_a, transpose_b=True), tf.matmul(x, split_b, transpose_b=True)

        if distribution == 'logistic':
            alpha = tf.math.sigmoid(b - a) * steepness

        elif distribution == 'logistic_phi':
            alpha = tf.math.sigmoid((b - a) * steepness / tf.pow(tf.math.abs(a - b) + 1.e-10, art_lambda, name=None))

        elif distribution == 'gaussian':
            v = b - a
            ncdf = NormalCDF(1 / steepness)
            alpha = ncdf.call(v)

        elif distribution == 'reciprocal':
            v = steepness * (b - a)
            alpha = 0.5 * (v / (2 + tf.abs(v)) + 1)

        elif distribution == 'cauchy':
            v = steepness * (b - a)
            alpha = 1 / math.pi * tf.math.atan(v) + .5
            alpha = tf.cast(alpha, dtype=x.dtype)

        elif distribution == 'optimal':
            v = steepness * (b - a)
            alpha = s_best(v)

        else:
            raise NotImplementedError('softmax method `{}` unknown'.format(distribution))

        aX = tf.matmul(X, tf.transpose(split_a))
        bX = tf.matmul(X, tf.transpose(split_b))
        w_min = tf.expand_dims(alpha, axis=-2) * aX + tf.expand_dims(1 - alpha, axis=-2) * bX
        w_max = tf.expand_dims(1 - alpha, axis=-2) * aX + tf.expand_dims(alpha, axis=-2) * bX
        X = tf.matmul(w_max, tf.expand_dims(tf.transpose(combine_max), axis=-3)) + \
            tf.matmul(w_min, tf.expand_dims(tf.transpose(combine_min), axis=-3))
        x = tf.matmul(alpha * a + (1 - alpha) * b, tf.transpose(combine_min)) + tf.matmul((1 - alpha) * a + alpha * b,
                                                                                          tf.transpose(combine_max))
    return x, X


def sort(
        sorting_network: SORTING_NETWORK_TYPE,
        vectors,
        steepness: float = 10.0,
        art_lambda: float = 0.25,
        distribution: str = 'cauchy'):
    """Sort a matrix along axis 1 using differentiable sorting networks. Return the permutation matrix.
    Positional arguments:
    sorting_network
    vectors -- the matrix to sort along axis 1; sorted in-place
    Keyword arguments:
    steepness -- relevant for sigmoid and sigmoid_phi interpolation (default 10.0)
    art_lambda -- relevant for logistic_phi interpolation (default 0.25)
    distribution -- how to interpolate when swapping two numbers; (default 'cauchy')
    """
    assert sorting_network[0][0].device == vectors.device, (
        f"The sorting network is on device {sorting_network[0][0].device} while the vectors are on device"
        f" {vectors.device}, but they both need to be on the same device."
    )
    return execute_sort(
        sorting_network=sorting_network,
        vectors=vectors,
        steepness=steepness,
        art_lambda=art_lambda,
        distribution=distribution
    )

class DiffSortNet:
    """Sort a matrix along axis 1 using differentiable sorting networks. Return the permutation matrix.
    Positional arguments:
    sorting_network_type -- which sorting network to use for sorting.
    vectors -- the matrix to sort along axis 1; sorted in-place
    Keyword arguments:
    steepness -- relevant for sigmoid and sigmoid_phi interpolation (default 10.0)
    art_lambda -- relevant for sigmoid_phi interpolation (default 0.25)
    interpolation_type -- how to interpolate when swapping two numbers; supported: `logistic`, `logistic_phi`,
                 (default 'logistic_phi')
    """
    def __init__(
        self,
        sorting_network_type: str,
        size: int,
        device: str = '',
        steepness: float = 10.0,
        art_lambda: float = 0.25,
        interpolation_type: str = None,
        distribution: str = 'cauchy',
    ):
        self.sorting_network_type = sorting_network_type
        self.size = size
        self.sorting_network = get_sorting_network(sorting_network_type, size, device)

        if interpolation_type is not None:
            assert distribution is None or distribution == 'cauchy' or distribution == interpolation_type, (
                'Two different distributions have been set (distribution={} and interpolation_type={}); however, '
                'they have the same interpretation and interpolation_type is a deprecated argument'.format(
                    distribution, interpolation_type
                )
            )
            distribution = interpolation_type

        self.steepness = steepness
        self.art_lambda = art_lambda
        self.distribution = distribution

    def forward(self, vectors, with_sorted_results=False):
        assert len(vectors.shape) == 2
        assert vectors.shape[1] == self.size
        sorted_rst, permutation_matrix = sort(
            self.sorting_network, vectors, self.steepness, self.art_lambda, self.distribution
        )
        if with_sorted_results:
            return sorted_rst, permutation_matrix
        return permutation_matrix

    @staticmethod
    def get_default_config():
        config = {}
        config['size'] = 10
        config['device'] = ''
        config['steepness'] = 10.0
        config['art_lambda'] = 0.25
        config['interpolation_type'] = None
        config['distribution'] = 'cauchy'
        return config

    @staticmethod
    def get_supported_distributions():
        return ['logistic', 'logistic_phi', 'gaussian', 'reciprocal', 'cauchy', 'optimal']


def bl_matmul(A, B):
  return tf.einsum('mij,jk->mik', A, B)

# s: M x n
# neuralsort(s): M x n x n
def neuralsort(s, tau=1):
    s = tf.expand_dims(s,axis=-1)
    A_s = s - tf.transpose(s, perm=[0, 2, 1])
    A_s = tf.abs(A_s)
    # As_ij = |s_i - s_j|

    n = tf.shape(s)[1]
    #n=s.shape[1]
    one = tf.ones((n, 1), dtype=tf.float32)

    B = bl_matmul(A_s, one @ tf.transpose(one))
    # B_:k = (A_s)(one)

    K = tf.range(n) + 1
    # K_k = k

    C = bl_matmul(
        s, tf.expand_dims(tf.cast(n + 1 - 2 * K, dtype=tf.float32), 0)
    )
    # C_:k = (n + 1 - 2k)s

    P = tf.transpose(C - B, perm=[0, 2, 1])
    # P_k: = (n + 1 - 2k)s - (A_s)(one)

    P = tf.nn.softmax(P / tau, -1)
    # P_k: = softmax( ((n + 1 - 2k)s - (A_s)(one)) / tau )

    return P


# s: M x n
# neuralsort(s): M x n x n
def neuralsort_reverse(s, tau=1):
    s = tf.expand_dims(s,axis=-1)
    A_s = s - tf.transpose(s, perm=[0, 2, 1])
    A_s = tf.abs(A_s)
    # As_ij = |s_i - s_j|

    n = tf.shape(s)[1]
    one = tf.ones((n, 1), dtype=tf.float32)

    B = bl_matmul(A_s, one @ tf.transpose(one))
    # B_:k = (A_s)(one)

    K = tf.range(n) + 1
    # K_k = k

    C = bl_matmul(
        s, tf.expand_dims(tf.cast(n + 1 - 2 * K, dtype=tf.float32), 0)
    )
    # C_:k = (n + 1 - 2k)s

    P = tf.transpose(C - B, perm=[0, 2, 1])
    # P_k: = (n + 1 - 2k)s - (A_s)(one)

    P = tf.nn.softmax(P / tau, -2)
    # P_k: = softmax( ((n + 1 - 2k)s - (A_s)(one)) / tau )

    return P

def soft_sort_row(s, tau):
    s=tf.expand_dims(s,axis=-1)
    s_sorted = tf.sort(s, direction='DESCENDING', axis=1)
    pairwise_distances = -tf.abs(tf.transpose(s, perm=[0, 2, 1]) - s_sorted)
    P_hat = tf.nn.softmax(pairwise_distances / tau, -1)
    return P_hat

def soft_sort_col(s, tau):
    s=tf.expand_dims(s,axis=-1)
    s_sorted = tf.sort(s, direction='DESCENDING', axis=1)
    pairwise_distances = -tf.abs(tf.transpose(s, perm=[0, 2, 1]) - s_sorted)
    P_hat = tf.nn.softmax(pairwise_distances / tau, -2)
    return P_hat

class NeuralSortNet:
    def __init__(self, tau, col_unimodal=False, descending=True):
        self.tau=tau
        self.col_unimodal=col_unimodal
        self.descending=descending
        if not self.col_unimodal:
            self.net = neuralsort
        else:
            self.net = neuralsort_reverse

    def forward(self, x):
        if self.descending:
            return self.net(x,self.tau)
        else:
            return self.net(x,self.tau)[::-1]

    @staticmethod
    def get_default_config():
        config = {}
        config['tau'] = 1
        return config

class SoftSortNet:
    def __init__(self, tau, col_unimodal=False, descending=True):
        self.tau=tau
        self.col_unimodal=col_unimodal
        self.descending=descending
        if not self.col_unimodal:
            self.net = soft_sort_row
        else:
            self.net = soft_sort_col

    def forward(self, x):
        if self.descending:
            return self.net(x,self.tau)
        else:
            return self.net(x,self.tau)[::-1]

    @staticmethod
    def get_default_config():
        config = {}
        config['tau'] = 1
        return config


if __name__ == '__main__':
    a = tf.convert_to_tensor([[4,3,1,2]],dtype=tf.float32)
    s = NeuralSortNet(1, col_unimodal=False, descending=True)
    b = s.forward(a)
    print(b)
    print(tf.reduce_sum(b,axis=-2))
    #with tf.Session() as sess:
    print(b/tf.expand_dims(tf.reduce_sum(b,axis=-2),axis=-1))