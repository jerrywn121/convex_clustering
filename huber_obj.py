import numpy as np
from itertools import combinations


def huber(x, delta):
    '''
    Args:
        x: input that has been norm2ed (n*(n-1)/2,)
        delta: threshold
    Output:
        (n*(n-1)/2,)
    '''
    return np.where(x > delta ** 2, np.sqrt(x) - delta / 2, x / (2 * delta))


def pair_col_diff_norm2(x, idx):
    '''
    compute norm2 of pairwise column difference
    Args:
        x: (d, n)
        idx: (n*(n - 1)/2, 2), used to indexing pairwise 
             column combinations
    Output:
        (n*(n-1)/2,)
    '''
    x = x[:, idx]  # (d, n*(n - 1)/2, 2)
    x = np.diff(x, axis=-1).squeeze()  # (d, n*(n-1)/2)
    x = np.sum(x ** 2, axis=0)  # (n*(n-1)/2,)
    return x


def pair_col_diff_sum(x, t, idx):
    '''
    compute sum of product of pairwise column difference of x and t
    Args:
        x: (d, n)
        t: (d, n)
        idx: (n*(n - 1)/2, 2), used to indexing pairwise column combinations
    Output:
        (n*(n-1)/2,)
    '''
    x = np.diff(x[:, idx], axis=-1).squeeze()  # (d, n*(n-1)/2)
    t = np.diff(t[:, idx], axis=-1).squeeze()  # (d, n*(n-1)/2)
    return np.sum(x * t, axis=0)  # (n*(n-1)/2,)


class OBJ:
    def __init__(self, d, n, delta):
        '''
        a: training data samples of shape (d, n)
        '''
        self.d = d
        self.n = n
        self.delta = delta
        self.idx = np.array(list(combinations(list(range(n)), 2)))
        self.triu_idx = np.triu_indices(self.n, 1)

    def __call__(self, x, a, lamb):
        '''
        Args:
            x: (d, n)
            a: (d, n)
            lamb: control effect of regularization
        Output:
            scalar
        '''
        v = np.sum((x - a) ** 2) / 2
        v += lamb * np.sum(huber(pair_col_diff_norm2(x, self.idx), self.delta))
        return v

    def grad(self, x, a, lamb):
        '''
        gradient
        Output:
            (d, n)
        '''
        g = x - a
        diff_norm2 = pair_col_diff_norm2(x, self.idx)  # (n*(n-1)/2,)
        tmp = np.zeros((self.n, self.n))
        tmp[self.triu_idx] = diff_norm2
        tmp += tmp.T  # (n, n)
        mask = (tmp > self.delta ** 2)
        tmp = np.where(mask,
                       np.divide(1, np.sqrt(tmp), where=mask),
                       0)
        x = x.T
        g = g + lamb * (tmp.sum(axis=1, keepdims=True) * x - tmp @ x).T
        tmp = 1 - mask
        g = g + lamb * (tmp.sum(axis=1, keepdims=True) * x - tmp @ x).T / self.delta
        return g.flatten()

    def hessiant(self, x, t, lamb):
        '''
        returns the result of hessian matrix dot product a vector t
        Args:
            t: (d, n)
        Output:
            (d, n)
        '''
        ht = 0
        ht += t
        diff_norm2 = pair_col_diff_norm2(x, self.idx)  # (n*(n-1)/2,)
        diff_sum = pair_col_diff_sum(x, t, self.idx)
        tmp = np.zeros((self.n, self.n))
        tmp[self.triu_idx] = diff_norm2
        tmp += tmp.T

        mask = (tmp > self.delta ** 2)
        tmp = np.where(mask,
                       np.divide(1, np.sqrt(tmp), where=mask),
                       0)
        t = t.T
        x = x.T
        ht += (lamb * (tmp.sum(axis=1, keepdims=True) * t - tmp @ t).T)
        # tmp1 = np.where(tmp1 > 0, tmp1 ** 3, 0)
        tmp = tmp ** 3
        tmp[self.triu_idx] *= diff_sum
        tmp[(self.triu_idx[1], self.triu_idx[0])] *= diff_sum
        ht -= lamb * (tmp.sum(axis=1, keepdims=True) * x - tmp @ x).T

        tmp = 1 - mask
        ht += (lamb * (tmp.sum(axis=1, keepdims=True) * t - tmp @ t).T / self.delta)
        return ht.flatten()





