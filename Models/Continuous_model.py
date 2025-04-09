""""
The original code comes from https://github.com/xunzheng/notears/blob/master/notears/utils.py.
"""

import numpy as np
from scipy.special import expit as sigmoid
import igraph as ig
import random
import pandas as pd


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

class Continuous_Model:
    # def __init__(self, bool_radndom_DAG: bool= False, DAG: pd.DataFrame = None, num_nodes: int = 0,seed: int = 0):
    #     self.num_nodes = num_nodes
    # DAG 是一个行索引和列索引都存在的 DataFrame
    """
    effect_ranges: tuple, default ((-1.0, -0.5), (0.5, 1.0))
        The ranges of the weights in the weighted adjacency matrix.
        The first range is for negative weights, and the second range is for positive weights.
        The weights are sampled uniformly from these ranges.
    """
    def __init__(self, DAG: pd.DataFrame, function_type: str = 'linear', noise_type: str = 'gaussian', sample_size: int = 1000, **kwargs):
        self.kwargs = kwargs
        self.node_names = DAG.columns.tolist()
        self.num_nodes = len(DAG.columns)
        self.adj_matrix = DAG.to_numpy().astype(int)
        self.sample_size = sample_size
        self.function_type = function_type
        self.noise_type = noise_type
        self.bool_is_dag = self.is_dag()
        if not self.bool_is_dag:
            raise ValueError("The input graph is not a DAG.")


    def _read_information_DAG(self):
        """Read the information of the DAG."""
        print(f"Graph Information:")
        print(f"  - Number of nodes: {self.num_nodes}")
        print(f"  - Node names: {', '.join(self.node_names)}")
        print(f"  - Adjacency matrix shape: {self.adj_matrix.shape}")
        print(f"  - Adjacency matrix data type: {self.adj_matrix.dtype}")
        print(f"  - Is DAG: {'Yes' if self.bool_is_dag else 'No'}")
        print(f"  - Number of edges: {np.sum(self.adj_matrix)}")
        print(f"  - Adjacency matrix:\n{self.adj_matrix}")
        

    def is_dag(self) -> bool:
        """Check if the adjacency matrix is a directed acyclic graph (DAG)."""
        G = ig.Graph.Weighted_Adjacency(self.adj_matrix.tolist())
        return G.is_dag()

    # def is_topological_sort(self):
    #     """Check if the adjacency matrix is arranged in topological order."""
    #     visited = set()
    #     for i in range(self.num_nodes):
    #         for j in range(i + 1, self.num_nodes):
    #             if self.adj_matrix[j, i] == 1:  # If there's a backward edge
    #                 return False
    #         visited.add(i)
    #     return True

    def simulate_scm(self) -> pd.DataFrame:
        """Simulate samples from the SCM."""
        if self.function_type == 'linear':
            return self.simulate_linear_scm()
        elif self.function_type == 'nonlinear':
            return self.simulate_nonlinear_scm()
        else:
            raise ValueError(f"Unknown function type: {self.function_type}")
    
    def simulate_linear_scm(self) -> pd.DataFrame:

        """"Simulate the Wight Adjacency Matrix based on the Adjacency Matrix."""
        def simulate_parameter(B, w_ranges=((-2.0, -0.5), (0.5, 2.0))):
            """Simulate SCM parameters for a DAG.

            Args:
                B (np.ndarray): [d, d] binary adj matrix of DAG
                w_ranges (tuple): disjoint weight ranges

            Returns:
                W (np.ndarray): [d, d] weighted adj matrix of DAG
            """
            W = np.zeros(B.shape)
            S = np.random.randint(len(w_ranges), size=B.shape)  # S.shape = [d, d], S[i,j] = k means W[i,j] in k-th range, k = 0,1,...,len(w_ranges)-1
            for i, (low, high) in enumerate(w_ranges):
                U = np.random.uniform(low=low, high=high, size=B.shape)
                W += B * (S == i) * U  # W[a,b] = 0 if S[a,b] != i, else W[a,b] = U[a,b]
            return W
        
        """x = Parents_X @ W + z, where z is the noise term."""
        def _simulate_single_equation(Pa_X, w, scale, scm_type,n):
            """
            Pa_x: np.array, rows = samples, cols = parents of x, [n, num of parents]
            w: np.array, [num of parents, 1 ], weight vector
            scale: float, scale of noise
            n: int, num of samples
            sem_type: str, type of SEM, e.g., 'gaussian', 'exp', 'gumbel', 'uniform', 'logistic', 'poisson'
            """
            if scm_type == 'gaussian':
                z = np.random.normal(scale=scale, size=n)
                x = Pa_X @ w + z
            elif scm_type == 'exp':
                z = np.random.exponential(scale=scale, size=n)
                x = Pa_X @ w + z
            elif scm_type == 'gumbel':
                z = np.random.gumbel(scale=scale, size=n)
                x = Pa_X @ w + z
            elif scm_type == 'uniform':
                z = np.random.uniform(low=-scale, high=scale, size=n)
                x = Pa_X @ w + z
            elif scm_type == 'logistic':
                x = np.random.binomial(1, sigmoid(Pa_X @ w)) * 1.0
            elif scm_type == 'poisson':
                x = np.random.poisson(np.exp(Pa_X @ w)) * 1.0
            else:
                raise ValueError('unknown sem type')
            return x


        """
        scale_vec is the scale of the noise term for each node.
        If noise_scale is a scalar, then all nodes have the same scale.
        """
        noise_scale = self.kwargs.get('noise_scale')
        if  noise_scale is None:
            scale_vec = np.ones(self.num_nodes)
        elif np.isscalar(noise_scale):  # if noise_scale is a scalar
            scale_vec = noise_scale * np.ones(self.num_nodes)
        else:
            if len(noise_scale) != self.num_nodes:
                raise ValueError('noise scale must be a scalar or has length d')
            scale_vec = noise_scale


        effect_ranges = self.kwargs.get('effect_ranges', ((-1.0, -0.5), (0.5, 1.0)))
        Weight_matrix = simulate_parameter(self.adj_matrix, w_ranges=effect_ranges)


        G = ig.Graph.Weighted_Adjacency(Weight_matrix.tolist())
        ordered_vertices = G.topological_sorting()
        assert len(ordered_vertices) == self.num_nodes
        data_matrix = np.zeros([self.sample_size, self.num_nodes])
        for i in ordered_vertices:
            parents = G.neighbors(i, mode=ig.IN)
            data_matrix[:, i] = _simulate_single_equation(data_matrix[:, parents], Weight_matrix[parents, i], scale_vec[i], self.noise_type, self.sample_size)

        return pd.DataFrame(data_matrix, columns=self.node_names)



    def simulate_nonlinear_sem(self)-> pd.DataFrame:
        """Simulate samples from nonlinear SEM.

        Args:
            B (np.ndarray): [d, d] binary adj matrix of DAG
            n (int): num of samples
            sem_type (str): mlp, mim, gp, gp-add
            noise_scale (np.ndarray): scale parameter of additive noise, default all ones

        Returns:
            X (np.ndarray): [n, d] sample matrix
        """
        def _simulate_single_equation(X, scale,sem_type):
            """X: [n, num of parents], x: [n]"""
            z = np.random.normal(scale=scale, size=self.sample_size)
            pa_size = X.shape[1]
            if pa_size == 0:
                return z
            if sem_type == 'mlp':
                hidden = 100
                W1 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
                W1[np.random.rand(*W1.shape) < 0.5] *= -1
                W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
                W2[np.random.rand(hidden) < 0.5] *= -1
                x = sigmoid(X @ W1) @ W2 + z
            elif sem_type == 'mim':
                w1 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
                w1[np.random.rand(pa_size) < 0.5] *= -1
                w2 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
                w2[np.random.rand(pa_size) < 0.5] *= -1
                w3 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
                w3[np.random.rand(pa_size) < 0.5] *= -1
                x = np.tanh(X @ w1) + np.cos(X @ w2) + np.sin(X @ w3) + z
            elif sem_type == 'gp':
                from sklearn.gaussian_process import GaussianProcessRegressor
                gp = GaussianProcessRegressor()
                x = gp.sample_y(X, random_state=None).flatten() + z
            elif sem_type == 'gp-add':
                from sklearn.gaussian_process import GaussianProcessRegressor
                gp = GaussianProcessRegressor()
                x = sum([gp.sample_y(X[:, i, None], random_state=None).flatten()
                        for i in range(X.shape[1])]) + z
            else:
                raise ValueError('unknown sem type')
            return x

        sem_type = self.kwargs.get('sem_type', 'mlp')
        if sem_type not in ['mlp', 'mim', 'gp', 'gp-add']:
            raise ValueError('unknown sem type')
        noise_scale = self.kwargs.get('noise_scale')
        scale_vec = noise_scale if noise_scale else np.ones(self.num_nodes)
        data_matrix = np.zeros([self.sample_size, self.num_nodes])
        G = ig.Graph.Adjacency(self.adj_matrix.tolist())
        ordered_vertices = G.topological_sorting()
        assert len(ordered_vertices) == self.num_nodes
        for j in ordered_vertices:
            parents = G.neighbors(j, mode=ig.IN)
            data_matrix[:, j] = _simulate_single_equation(data_matrix[:, parents], scale_vec[j],sem_type)
        return pd.DataFrame(data_matrix, columns=self.node_names)






def simulate_dag(d:int, s0:int, graph_type):

    """    Simulate a random Directed Acyclic Graph (DAG) with a specified number of nodes and edges.
        This function generates a random DAG based on the specified graph type. It supports three types of graphs:
        - Erdos-Renyi (ER): A random graph where edges are added between nodes with a fixed probability.
        - Scale-Free (SF): A graph generated using the Barabasi-Albert model, where new nodes preferentially attach to existing nodes with higher degrees.
        - Bipartite (BP): A bipartite graph where nodes are divided into two disjoint sets, and edges only connect nodes from different sets.
        The function ensures that the generated graph is acyclic by applying a random permutation and retaining only the lower triangular part of the adjacency matrix.
            d (int): The number of nodes in the graph.
            s0 (int): The expected number of edges in the graph.
            graph_type (str): The type of graph to generate. Must be one of 'ER', 'SF', or 'BP'.
            np.ndarray: A binary adjacency matrix of shape [d, d] representing the generated DAG.
        Raises:
            ValueError: If an unknown graph type is provided.
        Notes:
            - The adjacency matrix is binary, where a value of 1 indicates the presence of a directed edge between two nodes.
            - The function uses the `igraph` library for graph generation and manipulation.
            - The output graph is guaranteed to be a DAG (Directed Acyclic Graph)."""
    """Simulate random DAG with some expected number of edges.

    Args:
        d (int): num of nodes
        s0 (int): expected num of edges
        graph_type (str): ER, SF, BP

    Returns:
        B (np.ndarray): [d, d] binary adj matrix of DAG
    """
    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if graph_type == 'ER':
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == 'SF':
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
        B = _graph_to_adjmat(G)
    elif graph_type == 'BP':
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * d)
        G = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)
    else:
        raise ValueError('unknown graph type')
    
    B_perm = _random_permutation(B)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    # topological order
    topological_sorting = ig.Graph.Adjacency(B_perm.tolist()).topological_sorting()
    B_perm = B_perm[topological_sorting, :][:, topological_sorting]
    return B_perm








if __name__ == '__main__':
    # Test the Continuous_Model class
    num_nodes = 5
    num_edges = 10
    DAG = simulate_dag(num_nodes, num_edges, 'ER')
    # print(DAG)
    num_samples = 1000
    # Convert the adjacency matrix into a pandas DataFrame and assign indices to its rows and columns, labeled as V1, V2, V3, ..., Vn
    DAG = pd.DataFrame(DAG, index=[f'V{i+1}' for i in range(num_nodes)], columns=[f'V{i+1}' for i in range(num_nodes)])
    model = Continuous_Model(DAG, function_type='linear', noise_type='gaussian', sample_size=num_samples)
    model._read_information_DAG()
    data = model.simulate_scm()
    print(data.head())