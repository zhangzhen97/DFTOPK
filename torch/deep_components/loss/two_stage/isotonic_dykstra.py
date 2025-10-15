import torch
import torch.nn.functional as F

def _cumsum_triu(x):
    """PyTorch implementation of triangular cumsum"""
    mask = torch.triu(torch.ones(x.shape, dtype=torch.bool, device=x.device))
    return torch.einsum('ij,jk->ik', x, mask.float())

def _jvp_isotonic_mask(solution, vector, eps=1e-4):
    """Jacobian-vector product for isotonic regression"""
    x = solution
    mask = F.pad(torch.abs(torch.diff(x)) <= eps, (1, 0), value=False)
    ar = torch.arange(x.size(0), device=x.device)
    
    inds_start = torch.where(mask == 0, ar, torch.tensor(float('inf'), device=x.device))
    inds_start = torch.sort(inds_start).values
    
    one_hot_start = F.one_hot(inds_start, num_classes=len(vector)).float()
    a = _cumsum_triu(one_hot_start)
    a = torch.cat([torch.diff(a.flip(0), axis=0).flip(0), a[-1].unsqueeze(0)], dim=0)
    return ((a.T * (a @ vector)) / (a.sum(1, keepdim=True) + 1e-8).sum(0))

def isotonic_dykstra_mag(s, w, l=1e-1, num_iter=500):
    """Weighted isotonic regression in PyTorch"""
    def f(v, u):
        d = v[1::2] - v[::2]
        s_num = (v * u)[::2] + (v * u)[1::2]
        s_den = u[::2] + u[1::2]
        
        mask = torch.repeat_interleave(d < 0, 2)
        mean = torch.repeat_interleave(s_num / s_den, 2)
        return v * mask + mean * (~mask)
    
    u = 1 + l * w
    
    def body_fn(vpq):
        xk, pk, qk = vpq
        yk = F.pad(f(xk[:-1] + pk[:-1], u[:-1]), (0, 1), value=xk[-1] + pk[-1])
        p = xk + pk - yk
        v = F.pad(f(yk[1:] + qk[1:], u[1:]), (1, 0), value=yk[0] + qk[0])
        q = yk + qk - v
        return v, p, q
    
    # Ensure odd length
    n = s.shape[0]
    if n % 2 == 0:
        minv = s.min().detach() - 1
        s = F.pad(s, (0, 1), value=minv)
        u = F.pad(u, (0, 1), value=0.0)
    
    v = s.clone()
    p = torch.zeros_like(s)
    q = torch.zeros_like(s)
    vpq = (v, p, q)
    
    for _ in range(num_iter // 2):
        vpq = body_fn(vpq)
    
    sol = vpq[0]
    return sol if n % 2 != 0 else sol[:-1]

def _jvp_isotonic_mag(solution, vector, w, l, eps=1e-4):
    """Jacobian-vector product for weighted isotonic regression"""
    x = solution
    mask = F.pad(torch.abs(torch.diff(x)) <= eps, (1, 0), value=False)
    ar = torch.arange(x.size(0), device=x.device)
    
    inds_start = torch.where(mask == 0, ar, torch.tensor(float('inf'), device=x.device))
    inds_start = torch.sort(inds_start).values
    
    u = 1 + l * w
    one_hot_start = F.one_hot(inds_start, num_classes=len(vector)).float()
    a = _cumsum_triu(one_hot_start)
    a = torch.cat([torch.diff(a.flip(0), axis=0).flip(0), a[-1].unsqueeze(0)], dim=0)
    return ((a.T * (a @ (vector * u))) / ((a * u).sum(1, keepdim=True) + 1e-8)).sum(0)


def isotonic_dykstra_mask(s, num_iter=500):
    """PyTorch implementation of isotonic regression using Dykstra's projection algorithm.
    Now supports 2D input (batch processing).
    
    Args:
        s: input tensor of shape (batch_size, n) or (n,)
        num_iter: number of iterations
        
    Returns:
        sol: solution tensor of same shape as s
    """
    original_shape = s.shape
    if s.dim() == 1:
        s = s.unsqueeze(0)  # Add batch dimension
        
    batch_size, n = s.shape
    
    def f(v):
        # v shape: (batch_size, n)
        d = v[:, 1::2] - v[:, ::2]  # differences
        a = v[:, ::2] + v[:, 1::2]  # sums
        
        mask = torch.repeat_interleave(d < 0, 2, dim=1)
        mean = torch.repeat_interleave(a / 2.0, 2, dim=1)
        return v * mask + mean * (~mask)
    
    def body_fn(vpq):
        xk, pk, qk = vpq

        yk = F.pad(f(xk[:, :-1] + pk[:, :-1]), (0, 1))
        yk[:, -1] = (xk[:, -1] + pk[:, -1]).squeeze()
        
        p = xk + pk - yk
        
        v = F.pad(f(yk[:, 1:] + qk[:, 1:]), (1, 0))
        v[:, 0] = (yk[:, 0] + qk[:, 0]).squeeze()
        
        q = yk + qk - v
        return v, p, q
    
    # Ensure odd length
    if n % 2 == 0:
        minv = s.min(dim=1, keepdim=True)[0] - 1
        s = F.pad(s, (0, 1))
        s[:, -1] = minv.squeeze()
    
    v = s.clone()
    p = torch.zeros_like(s)
    q = torch.zeros_like(s)
    vpq = (v, p, q)
    
    for _ in range(num_iter // 2):
        vpq = body_fn(vpq)
    
    sol = vpq[0]
    if n % 2 != 0:
        result = sol
    else:
        result = sol[:, :-1]
    
    if len(original_shape) == 1:
        return result.squeeze(0)
    return result