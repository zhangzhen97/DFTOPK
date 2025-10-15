import torch.nn.functional as F
from deep_components.loss.two_stage.isotonic_dykstra import isotonic_dykstra_mask
import torch
def sparse_soft_topk_mask_dykstra(x, k, l=1e-1, num_iter=500):
    original_shape = x.shape
    if x.dim() == 1:
        x = x.unsqueeze(0) 
        
    batch_size, n = x.shape
    
    with torch.no_grad():
        perm = torch.argsort(-x, dim=1)
        P = F.one_hot(perm, n).float()
    
    s = torch.matmul(P, x.unsqueeze(-1)).squeeze(-1)
    s_w = s - l * F.pad(torch.ones((batch_size, k), device=x.device), (0, n - k))
    out_dykstra = isotonic_dykstra_mask(s_w, num_iter)
    out = (s - out_dykstra) / l
    result = torch.matmul(P.transpose(1, 2), out.unsqueeze(-1)).squeeze(-1)
    
    if len(original_shape) == 1:
        return result.squeeze(0)
    return result

def sparse_soft_topk_mag_dykstra(x, k, l=1e-1, num_iter=500):
    original_shape = x.shape
    if x.dim() == 1:
        x = x.unsqueeze(0)
        
    batch_size, n = x.shape
    
    with torch.no_grad():
        perm = torch.argsort(-torch.abs(x), dim=1)
        P = F.one_hot(perm, n).float()
    
    s = torch.matmul(P, torch.abs(x).unsqueeze(-1)).squeeze(-1)
    w = F.pad(torch.ones((batch_size, k), device=x.device), (0, n - k))
    adjusted_s = s / (1 + l * w)
    out_dykstra = isotonic_dykstra_mask(adjusted_s, num_iter)
    out = (s - out_dykstra) / l
    perm_out = torch.matmul(P.transpose(1, 2), out.unsqueeze(-1)).squeeze(-1)
    result = torch.sign(x) * perm_out * (1 + l)
    
    if len(original_shape) == 1:
        return result.squeeze(0)
    return result

def hard_topk_mask(x, k):
    if x.dim() == 1:
        x = x.unsqueeze(0)
        
    values, indices = torch.topk(x, k, dim=1)
    result = F.one_hot(indices, x.shape[-1]).sum(dim=1).float()
    
    if x.dim() == 1:
        return result.squeeze(0)
    return result

def hard_topk_mag(x, k):
    return x * hard_topk_mask(torch.abs(x), k)
