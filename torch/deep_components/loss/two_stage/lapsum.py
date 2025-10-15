from math import log
import time
import numpy as np
import torch

class LogSoftTopK(torch.autograd.Function):

    @staticmethod
    def _solve(s, t, a, b, e):
        z = torch.abs(e) + torch.sqrt(e**2 + a * b * torch.exp(s - t))
        ab = torch.where(e > 0, a, b)

        return torch.where(
            e > 0, t + torch.log(z) - torch.log(ab), s - torch.log(z) + torch.log(ab)
        )

    @staticmethod
    def forward(ctx, r, k, alpha, descending=False):
        # Sprawdzenie wymiarów
        assert r.shape[0] == k.shape[0], "k must have same batch size as r"

        batch_size, num_dim = r.shape
        x = torch.empty_like(r, requires_grad=False)

        def finding_b():
            scaled = torch.sort(r, dim=1)[0]
            scaled.div_(alpha)

            eB = torch.logcumsumexp(scaled, dim=1)
            eB.sub_(scaled).exp_()

            torch.neg(scaled, out=x)
            eA = torch.flip(x, dims=(1,))
            torch.logcumsumexp(eA, dim=1, out=x)
            idx = torch.arange(start=num_dim - 1, end=-1, step=-1, device=x.device)
            torch.index_select(x, 1, idx, out=eA)
            eA.add_(scaled).exp_()

            row = torch.arange(1, 2 * num_dim + 1, 2, device=r.device)

            torch.add(torch.add(eA, eB, alpha=-1, out=x), row.view(1, -1), out=x)

            w = (k if descending else num_dim - k).unsqueeze(1)
            i = torch.searchsorted(x, 2 * w)
            m = torch.clamp(i - 1, 0, num_dim - 1)
            n = torch.clamp(i, 0, num_dim - 1)

            b = LogSoftTopK._solve(
                scaled.gather(1, m),
                scaled.gather(1, n),
                torch.where(i < num_dim, eA.gather(1, n), 0),
                torch.where(i > 0, eB.gather(1, m), 0),
                w - i,
            )
            return b

        b = finding_b()

        sign = -1 if descending else 1

        torch.div(r, alpha * sign, out=x)
        x.sub_(sign * b)

        sign_x = x > 0
        qx = torch.relu(x).neg_().exp_().mul_(-0.5).add_(1)

        ctx.save_for_backward(x, qx, r)
        ctx.alpha = alpha
        ctx.sign = sign

        log_p = torch.where(sign_x, torch.log(qx), x.sub(log(2)))
        return log_p

    @staticmethod
    def backward(ctx, grad_output):
        x, qx, r = ctx.saved_tensors
        alpha = ctx.alpha
        sign = ctx.sign

        x.abs_().neg_()
        grad_r = torch.softmax(x, dim=1)
        x.exp_()
        grad_k = torch.sum(x, dim=1).mul_(0.5)

        qx.reciprocal_().sub_(1)
        qx.mul_(grad_output)  # wgrad

        wsum = qx.sum(dim=1, keepdim=True)

        # Gradients
        grad_k.reciprocal_().mul_(wsum.squeeze(1)).mul_(abs(sign))
        grad_r.mul_(wsum).sub_(qx).mul_(-sign / alpha)

        x.copy_(r).mul_(grad_r)
        grad_alpha = torch.sum(x).div_(-alpha)

        return grad_r, grad_k, grad_alpha, None

def log_soft_top_k(r, k, alpha, descending=False):
    return LogSoftTopK.apply(r, k, alpha, descending)

def numerical_vjp(x, k, alpha, descending, v, h=1e-5):
    grad_approx = torch.zeros_like(x)
    for i in range(x.numel()):
        e = torch.zeros_like(x).view(-1)
        e[i] = h  # Perturb one dimension at a time
        e = e.view_as(x)  # Reshape back to original shape

        grad_approx.view(-1)[i] = torch.dot(
            v.flatten(),
            (
                log_soft_top_k(x + e, k, alpha, descending)
                - log_soft_top_k(x - e, k, alpha, descending)
            ).flatten(),
        ) / (2 * h)
    return grad_approx

def check_value(x, v, text):
    assert x.shape == v.shape, f"Shape mismatch: {x.shape} vs {v.shape}"

    def fun():
        if isinstance(x, torch.Tensor):
            return torch.allclose, torch.linalg.norm
        else:
            return np.allclose, np.linalg.norm

    function, dist = fun()
    check = None
    for tol_exp in range(-15, 0):
        if function(x, v, rtol=1e-05, atol=10**tol_exp):
            check = f"Error within atol=1e{tol_exp}"
            break
    if check:
        print(f"✅ - {text} ({check})")
    else:
        print(f"❌ - {text} [dist: {dist(x - v):.4f}]")
        print(f"Expected: {v}")
        print(f"Got: {x}")

def print_time_stats(times, name):
    if not times:
        return
    avg = sum(times) / len(times)
    min_t = min(times)
    max_t = max(times)
    print(f"\n{name} time stats (seconds):")
    print(f"\033[0;1;35m  Average: {avg:.4f}\033[0m")
    print(f"  Min:     {min_t:.4f}")
    print(f"  Max:     {max_t:.4f}")
    print(f"  All times: {[f'{t:.4f}' for t in times]}")

class SoftTopK(torch.autograd.Function):
    @staticmethod
    def _solve(s, t, a, b, e):
        z = torch.abs(e) + torch.sqrt(e**2 + a * b * torch.exp(s - t))
        ab = torch.where(e > 0, a, b)

        return torch.where(
            e > 0, t + torch.log(z) - torch.log(ab), s - torch.log(z) + torch.log(ab)
        )

    @staticmethod
    def forward(ctx, r, k, alpha, descending=False):
        assert r.shape[0] == k.shape[0], "k must have same batch size as r"

        batch_size, num_dim = r.shape
        x = torch.empty_like(r, requires_grad=False)

        def finding_b():
            scaled = torch.sort(r, dim=1)[0]
            scaled.div_(alpha)

            eB = torch.logcumsumexp(scaled, dim=1)
            eB.sub_(scaled).exp_()

            torch.neg(scaled, out=x)
            eA = torch.flip(x, dims=(1,))
            torch.logcumsumexp(eA, dim=1, out=x)
            idx = torch.arange(start=num_dim - 1, end=-1, step=-1, device=x.device)
            torch.index_select(x, 1, idx, out=eA)
            eA.add_(scaled).exp_()

            row = torch.arange(1, 2 * num_dim + 1, 2, device=r.device)

            torch.add(torch.add(eA, eB, alpha=-1, out=x), row.view(1, -1), out=x)

            w = (k if descending else num_dim - k).unsqueeze(1)
            i = torch.searchsorted(x, 2 * w)
            m = torch.clamp(i - 1, 0, num_dim - 1)
            n = torch.clamp(i, 0, num_dim - 1)

            b = SoftTopK._solve(
                scaled.gather(1, m),
                scaled.gather(1, n),
                torch.where(i < num_dim, eA.gather(1, n), 0),
                torch.where(i > 0, eB.gather(1, m), 0),
                w - i,
            )
            return b

        b = finding_b()

        sign = -1 if descending else 1
        torch.div(r, alpha * sign, out=x)
        x.sub_(sign * b)

        sign_x = x > 0
        p = torch.abs(x)
        p.neg_().exp_().mul_(0.5)

        inv_alpha = -sign / alpha
        S = torch.sum(p, dim=1, keepdim=True).mul_(inv_alpha)

        torch.where(sign_x, 1 - p, p, out=p)

        ctx.save_for_backward(r, x, S)
        ctx.alpha = alpha
        return p

    @staticmethod
    def backward(ctx, grad_output):
        r, x, S = ctx.saved_tensors
        alpha = ctx.alpha

        x.abs_().neg_()
        q = torch.softmax(x, dim=1)

        torch.mul(q, grad_output, out=x)
        grad_k = x.sum(dim=1, keepdim=True)

        grad_r = grad_k - grad_output
        grad_r.mul_(q).mul_(S)

        q.mul_(r)
        x.mul_(S / alpha)  # grad_alpha = (S / alpha) * x
        r.sub_(q.sum(dim=1, keepdim=True))
        x.mul_(r) 
        return grad_r, None, None, None

def soft_top_k(r, k, alpha, descending=False):
    return SoftTopK.apply(r, k, alpha, descending)

def soft_top_k_autograd(r, k, alpha, descending=False):
    assert r.shape[0] == k.shape[0], "k must have same batch size as r"
    def _solve(s, t, a, b, e):
        z = torch.abs(e) + torch.sqrt(e**2 + a * b * torch.exp(s - t))
        ab = torch.where(e > 0, a, b)
        epsilon = 1e-9
        return torch.where(
            e > 0, 
            t + torch.log(z + epsilon) - torch.log(ab + epsilon), 
            s - torch.log(z + epsilon) + torch.log(ab + epsilon)
        )
    batch_size, num_dim = r.shape
    scaled = torch.sort(r, dim=1)[0]
    scaled = scaled / alpha

    eB = torch.logcumsumexp(scaled, dim=1)
    eB = torch.exp(eB - scaled)
    neg_scaled = -scaled 
    eA = torch.flip(neg_scaled, dims=(1,))
    eA = torch.logcumsumexp(eA, dim=1)
    idx = torch.arange(start=num_dim - 1, end=-1, step=-1, device=eA.device)
    eA = torch.index_select(eA, 1, idx)
    eA = torch.exp(eA + scaled)
    row = torch.arange(1, 2 * num_dim + 1, 2, device=r.device)
    x_for_search = eA - eB + row.view(1, -1)
    w = (k if descending else num_dim - k).unsqueeze(1)
    i = torch.searchsorted(x_for_search, 2 * w)
    m = torch.clamp(i - 1, 0, num_dim - 1)
    n = torch.clamp(i, 0, num_dim - 1)
    b = _solve(
        torch.gather(scaled, 1, m),
        torch.gather(scaled, 1, n),
        torch.where(i < num_dim, torch.gather(eA, 1, n), torch.tensor(0.0, device=r.device)),
        torch.where(i > 0, torch.gather(eB, 1, m), torch.tensor(0.0, device=r.device)),
        w - i.float(), 
    )
    sign = -1.0 if descending else 1.0
    x = r / (alpha * sign)
    x = x - (sign * b)

    sign_x = x > 0
    p_abs = torch.abs(x)
    p = torch.exp(-p_abs) * 0.5
    p_final = torch.where(sign_x, 1 - p, p)
    return p_final