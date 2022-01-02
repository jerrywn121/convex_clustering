import numpy as np
from utils import norm2, norm
import time


def accelerated_gradient_method(grad, x0, alpha, tol, display_interval=0):
    '''
    grad = lambda x: obj.grad(x, a, lamb)
    '''
    start_time = time.time()
    num_iter = 0
    x = x0
    x_prev = x
    g = grad(x)
    ng = norm(g)
    t_prev = 1
    t = 1
    beta = (t_prev - 1) / t
    ngs = [ng]
    while ng > tol:
        if display_interval > 0 and num_iter % display_interval == 0:
            print(f"num_iter: {num_iter}, ng: {ng}")
        y = x + beta * (x - x_prev)
        x, x_prev = y - alpha * grad(y), x
        num_iter += 1
        g = grad(x)
        ng = norm(g)
        ngs.append(ng)
        t, t_prev = (1 + np.sqrt(1 + 4 * t ** 2)) / 2, t
        beta = (t_prev - 1) / t
    return x, np.array(ngs), time.time() - start_time


def conjuate_gradient(grad, hessiant, x, v0, r0, n2r0, tol, max_iter):
    '''
    r0 = grad(x_k)
    n2r0: norm2 of r0
    '''
    assert tol > 0
    num_iter = 0
    v = v0
    r = r0
    n2r = n2r0
    p = -r
    while num_iter < max_iter:
        Ap = hessiant(x, p)
        pAp = np.sum(p * Ap)
        if pAp <= 0:
            if num_iter == 0:
                return -grad(x)
            else:
                return v
        else:
            sigma = n2r / pAp
            v = v + sigma * p
            r = r + sigma * Ap
            n2r, n2r_prev = norm2(r), n2r
            num_iter += 1
        if np.sqrt(n2r) < tol:
            return v
        beta = n2r / n2r_prev
        # beta = norm2(r) / norm2(r_prev)
        p = -r + beta * p
    return v


def newton_cg(f, grad, hessiant, x0, v0, tol, max_iter, cg_max, s, sigma, gamma, display_interval=0):
    '''
    f = lambda x: obj(x, a, lamb):
        output scalar
    grad = lambda x: obj.grad(x, a, lamb):
        output (n*d,)
    hessiant = lambda x, t: obj.hessiant(x, t, lamb): hessian evaluated at x dot t:
        output (n*d,)
    '''
    start_time = time.time()
    num_iter = 0
    x = x0
    g = grad(x)
    ng = norm(g)
    ngs = [ng]
    while ng > tol and num_iter < max_iter:
        if display_interval > 0 and num_iter % display_interval == 0:
            print(f"num_iter: {num_iter}, ng: {ng}")
        d = conjuate_gradient(grad, hessiant, x, v0=0, r0=g,
                              n2r0=(ng ** 2), tol=min(1, ng**0.1) * ng,
                              max_iter=cg_max)
        alpha = backtracking(x, f, f(x), g, d, s, sigma, gamma)
        x = x + alpha * d
        g = grad(x)
        ng = norm(g)
        ngs.append(ng)
        num_iter += 1

    return x, np.array(ngs), time.time() - start_time


def backtracking(x, obj, fx, g, d, s, sigma, gamma):
    alpha = s
    while obj(x + alpha * d) - fx > gamma * alpha * np.sum(g * d):
        alpha = alpha * sigma
    return alpha


