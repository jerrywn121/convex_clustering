import numpy as np
from itertools import combinations
from huber_obj import pair_col_diff_norm2


def norm(x):
    return np.sqrt(np.sum(x**2))


def norm2(x):
    return np.sum(x**2)


def cluster(x, epsilon):
    '''
    cluster n points according to x, points i and j are in the
    same group if ||xi - xj|| < epsilon
    Args:
        x: (d, n) stores each point's "centroid" in its columns
    Output:
        clustered: list containing the ID of the cluster of each point
        num_clusters: total number of clusters
    '''
    d, n = x.shape
    idx = np.array(list(combinations(list(range(n)), 2)))
    dist = np.sqrt(pair_col_diff_norm2(x, idx))  # (n, n)
    tmp = - np.ones((n, n))
    tmp[np.triu_indices(n, 1)] = dist
    idx = np.where((tmp < epsilon) & (tmp > 0))
    idx = list(zip(*idx))
    G = Graph(n)
    for i in idx:
        G.add_edge(i[0], i[1])
    CC = ConnectedComponent(G)
    num_clusters = CC.count
    clustered = CC.ID
    return clustered, num_clusters


class Graph:
    def __init__(self, V):
        self.V = V
        self.E = 0
        self.adj = [[] for _ in range(V)]

    def add_edge(self, v, w):
        self.adj[v].append(w)
        self.adj[w].append(v)
        self.E += 1


class ConnectedComponent:
    def __init__(self, G):
        self.G = G
        self.count = 0
        self.ID = [0] * G.V
        self.marked = [False] * G.V
        for v in range(G.V):
            if not self.marked[v]:
                self.dfs(G, v)
                self.count += 1

    def dfs(self, G, v):
        self.marked[v] = True
        self.ID[v] = self.count
        for w in G.adj[v]:
            if not self.marked[w]:
                self.dfs(G, w)




