import torch
import torch.nn.functional as F
import math
from typing import List, Tuple
import numpy as np
SORTING_NETWORK_TYPE = List[torch.tensor]
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
    def matrix_to_torch(m):
        return [[torch.from_numpy(matrix).float().to(device) for matrix in matrix_set] for matrix_set in m]

    if type == 'bitonic':
        return matrix_to_torch(bitonic_network(n))
    elif type == 'odd_even':
        return matrix_to_torch(odd_even_network(n))
    else:
        raise NotImplementedError('Sorting network `{}` unknown.'.format(type))

class DiffSortNet(torch.nn.Module):
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
        device: str = 'cpu',
        steepness: float = 10.0,
        art_lambda: float = 0.25,
        interpolation_type: str = None,
        distribution: str = 'cauchy',
    ):
        super(DiffSortNet, self).__init__()
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

    def forward(self, vectors):
        # assert len(vectors.shape) == 2
        # assert vectors.shape[1] == self.size
        return sort(
            self.sorting_network, vectors, self.steepness, self.art_lambda, self.distribution
        )

    @staticmethod
    def get_default_config():
        config = {}
        config['size'] = 40
        config['device'] = ''
        config['steepness'] = 10
        config['art_lambda'] = 0.25
        config['interpolation_type'] = None
        config['distribution'] = 'cauchy'
        return config

    @staticmethod
    def get_supported_distributions():
        return ['logistic', 'logistic_phi', 'gaussian', 'reciprocal', 'cauchy', 'optimal']

def s_best(x):
    return torch.clamp(x, -0.25, 0.25) + .5 + \
        ((x > 0.25).float() - (x < -0.25).float()) * (0.25 - 1/16/(x.abs()+1e-10))

def execute_sort(
        sorting_network,
        vectors,
        steepness=10.,
        art_lambda=0.25,
        distribution='cauchy'
):
    x = vectors
    X = torch.eye(vectors.shape[1], dtype=x.dtype, device=x.device).repeat(x.shape[0], 1, 1)

    for split_a, split_b, combine_min, combine_max in sorting_network:
        split_a = split_a.type(x.dtype)
        split_b = split_b.type(x.dtype)
        combine_min = combine_min.type(x.dtype)
        combine_max = combine_max.type(x.dtype)

        a, b = x @ split_a.T, x @ split_b.T

        new_type = torch.float32 if x.dtype == torch.float16 else x.dtype

        if distribution == 'logistic':
            alpha = torch.sigmoid((b-a).type(new_type) * steepness).type(x.dtype)

        elif distribution == 'logistic_phi':
            alpha = torch.sigmoid((b-a).type(new_type) * steepness / ((a-b).type(new_type).abs() + 1.e-10).pow(art_lambda)).type(x.dtype)

        elif distribution == 'gaussian':
            v = (b - a).type(new_type)
            alpha = NormalCDF.apply(v, 1 / steepness)
            alpha = alpha.type(x.dtype)

        elif distribution == 'reciprocal':
            v = steepness * (b - a).type(new_type)
            alpha = 0.5 * (v / (2 + v.abs()) + 1)
            alpha = alpha.type(x.dtype)

        elif distribution == 'cauchy':
            v = steepness * (b - a).type(new_type)
            alpha = 1 / math.pi * torch.atan(v) + .5
            alpha = alpha.type(x.dtype)

        elif distribution == 'optimal':
            v = steepness * (b - a).type(new_type)
            alpha = s_best(v)
            alpha = alpha.type(x.dtype)

        else:
            raise NotImplementedError('softmax method `{}` unknown'.format(distribution))

        aX = X @ split_a.T
        bX = X @ split_b.T
        w_min = alpha.unsqueeze(-2) * aX + (1-alpha).unsqueeze(-2) * bX
        w_max = (1-alpha).unsqueeze(-2) * aX + alpha.unsqueeze(-2) * bX
        X = (w_max @ combine_max.T.unsqueeze(-3)) + (w_min @ combine_min.T.unsqueeze(-3))
    return X 

def sort(
        sorting_network: SORTING_NETWORK_TYPE,
        vectors: torch.Tensor,
        steepness: float = 10.0,
        art_lambda: float = 0.25,
        distribution: str = 'cauchy'
) -> Tuple[torch.Tensor, torch.Tensor]:
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



def add_eb_v2(input, alpha, times=1, eb=0.1):    
    eb = eb / alpha * times
    sorted_value, idx = torch.sort(input, dim=-1)
    depth = idx.shape[1]
    mat = F.one_hot(idx, num_classes=depth).float()
    intervals = sorted_value - torch.cat([sorted_value[:, :1], sorted_value[:, :-1]], dim=1)
    intervals += (eb - intervals).relu().detach()
    cum_intervals = torch.cumsum(intervals, dim=1) - eb
    fixed_value = cum_intervals + sorted_value[:, 0:1]
    fixed_value = torch.bmm(fixed_value.unsqueeze(1), mat).squeeze(1)
    return fixed_value

class NormalCDF:
    """Normal CDF operation with custom gradient."""
    def __init__(self, sigma: float):
        self.sigma = sigma
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1.0 + torch.erf(x / (self.sigma * math.sqrt(2.0))))
    
    def backward(self, grad_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return grad_output * (1.0 / (self.sigma * math.sqrt(2.0 * math.pi))) * torch.exp(-0.5 * (x / self.sigma).pow(2))
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

def bl_matmul(A, B):
    return torch.einsum('mij,jk->mik', A, B)

# neuralsort(s): M x n x n
def neuralsort(s, tau=1):
    s = s.unsqueeze(-1)
    A_s = torch.abs(s - s.transpose(1, 2))
    # As_ij = |s_i - s_j|

    n = s.size(1)
    one = torch.ones((n, 1), dtype=torch.float32, device=s.device)

    B = bl_matmul(A_s, one @ one.T)
    # B_:k = (A_s)(one)

    K = torch.arange(n, device=s.device) + 1
    # K_k = k

    C = bl_matmul(s, (n + 1 - 2 * K).float().unsqueeze(0))
    # C_:k = (n + 1 - 2k)s

    P = (C - B).transpose(1, 2)
    # P_k: = (n + 1 - 2k)s - (A_s)(one)

    P = F.softmax(P / tau, dim=-1)
    # P_k: = softmax( ((n + 1 - 2k)s - (A_s)(one)) / tau )

    return P

def soft_sort(s, tau):
    s = s.unsqueeze(-1)
    s_sorted, _ = torch.sort(s, descending=True, dim=1)
    pairwise_distances = -torch.abs(s.transpose(1, 2) - s_sorted)
    P_hat = F.softmax(pairwise_distances / tau, dim=-1)
    return P_hat

class NeuralSortNet:
    def __init__(self, tau, descending=True):
        self.tau = tau
        self.descending = descending
        self.net = neuralsort

    def forward(self, x):
        if self.descending:
            return self.net(x, self.tau)
        else:
            return self.net(x, self.tau)[:, :, ::-1]

    @staticmethod
    def get_default_config():
        config = {}
        config['tau'] = 50
        return config

class SoftSortNet:
    def __init__(self, tau, descending=True):
        self.tau = tau
        self.descending = descending
        self.net = soft_sort

    def forward(self, x):
        if self.descending:
            return self.net(x, self.tau)
        else:
            return self.net(x, self.tau)[:, :, ::-1]

    @staticmethod
    def get_default_config():
        config = {}
        config['tau'] = 1
        return config

class SortNet:
    def __init__(self, sort_op, reverse=False, config=None):
        self.sort_op = sort_op
        self.config = config
        self.reverse = reverse
        if sort_op == 'neural_sort':
            self.net = NeuralSortNet(tau=self.config['tau'], descending=True)
        elif sort_op == 'soft_sort':
            self.net = SoftSortNet(tau=self.config['tau'], descending=True)
        elif sort_op == "diff_sort":
            self.net=DiffSortNet(sorting_network_type='odd_even',size=config['size'],
                    device=config['device'],steepness=config['steepness'],
                    art_lambda=config['art_lambda'],interpolation_type=config['interpolation_type'],
                    distribution=config['distribution'])
        else:
            raise NotImplementedError(f'[ERROR] sort_op `{sort_op}` unknown')

    def forward(self, x):
        if self.sort_op in ['diff_sort']:
            permutation_matrix = self.net.forward(-x)
        else:
            permutation_matrix = self.net.forward(x)
        if (self.reverse and self.sort_op in ['neural_sort', 'soft_sort']):
            return torch.flip(permutation_matrix, dims=[-2])
        else:
            return permutation_matrix

    @staticmethod
    def get_default_config(sort_op):
        if sort_op == 'neural_sort':
            return NeuralSortNet.get_default_config()
        elif sort_op == 'soft_sort':
            return SoftSortNet.get_default_config()
        elif sort_op == 'diff_sort':
            return DiffSortNet.get_default_config()
