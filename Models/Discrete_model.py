""""
Author: Zheng Li
Date: 2025-04-08
Description: This module implements a discrete model (conditional probability table) for simulating data from a directed acyclic graph (DAG).
"""

import numpy as np
from scipy.special import expit as sigmoid
import igraph as ig
import random
import pandas as pd
import networkx as nx
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling

# from Continuous_model import simulate_dag


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

class Discrete_Model:

    """
    Discrete_Model class for simulating data from a directed acyclic graph (DAG).
    """
    def __init__(self, DAG: pd.DataFrame = None, sample_size: int = 1000, num_values: int = 2, min_prob: float = 0.1,**kwargs):
        """
        Initialize the Discrete_Model.

        Args:
            DAG (pd.DataFrame): Adjacency matrix of the DAG.
            sample_size (int): Number of samples to generate.
            num_values (int): Number of discrete values each variable can take.
            min_prob (float): Minimum probability for any entry in the CPT(conditional probability table).
        """
        self.kwargs = kwargs
        self.node_names = DAG.columns.tolist()
        self.num_nodes = len(DAG.columns)
        self.adj_matrix = DAG.to_numpy().astype(int)
        self.nx_DAG = nx.from_pandas_adjacency(DAG, create_using=nx.DiGraph)
        self.sample_size = sample_size
        self.num_values = num_values
        self.min_prob = min_prob
        self.model = self._build_bayesian_model()
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
        print(f"  - Adjacency matrix:\n{self.adj_matrix}\n")
        print(f" - nx_DAG edges: \n{self.nx_DAG.edges()}")
        

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

    def _build_bayesian_model(self):
        """
        Build a BayesianModel using pgmpy based on the adjacency matrix.

        Returns:
            BayesianModel: The Bayesian network model.
        """

        
        # Create BayesianModel from adjacency matrix
        bayesian_model = BayesianNetwork(self.nx_DAG)
        # for i, node in enumerate(self.node_names):
        #     parents = [self.node_names[j] for j in range(self.num_nodes) if self.adj_matrix[j, i] == 1]
        #     for parent in parents:
        #         bayesian_model.add_edge(parent, node)

        # Generate random CPTs for each node
        for i, node in enumerate(self.node_names):
            parents = [self.node_names[j] for j in range(self.num_nodes) if self.adj_matrix[j, i] == 1]
            num_parent_states = self.num_values ** len(parents)

            # Generate random probabilities
            cpt = np.random.rand(num_parent_states, self.num_values)
            cpt = np.maximum(cpt, self.min_prob)  # Ensure minimum probability
            cpt /= cpt.sum(axis=1, keepdims=True)  # Normalize to sum to 1

            # Define TabularCPD
            cpd = TabularCPD(
                variable=node,
                variable_card=self.num_values,
                values=cpt.T,
                evidence=parents,
                evidence_card=[self.num_values] * len(parents)
            )
            bayesian_model.add_cpds(cpd)

        # Check if the model is valid
        assert bayesian_model.check_model(), "The Bayesian model is invalid."
        return bayesian_model

    def _read_bayesian_model(self):
        """
        Read the Bayesian model information.

        Returns:
            None
        """
        print("Bayesian Model Information:")
        print(f"  - Number of CPDs: {len(self.model.get_cpds())}")
        for cpd in self.model.get_cpds():
            print(f"    - {cpd}")

    def generate_data(self):
        """
        Generate discrete data using the Bayesian model.

        Returns:
            pd.DataFrame: A DataFrame containing the generated data.
        """
        if self.kwargs.get('show_progress') is None:
            show_progress = False
        else:
            show_progress = self.kwargs['show_progress']
        sampler = BayesianModelSampling(self.model)
        # Generate samples
        data = sampler.forward_sample(size=self.sample_size,show_progress=show_progress)
        return data












if __name__ == '__main__':
    # Test the Discrete_Model class
    num_nodes = 50
    num_edges = 120
    DAG = simulate_dag(num_nodes, num_edges, 'ER')
    """
    Example usage:
        V1 -> V2 -> V3
    """
    # DAG = np.array([[0, 1, 0],
    #                 [0, 0, 1],    
    #                 [0, 0, 0]])
    
    # print(DAG)
    num_samples = 1000
    # Convert the adjacency matrix into a pandas DataFrame and assign indices to its rows and columns, labeled as V1, V2, V3, ..., Vn
    DAG = pd.DataFrame(DAG, index=[f'V{i+1}' for i in range(num_nodes)], columns=[f'V{i+1}' for i in range(num_nodes)])
    model = Discrete_Model(DAG, sample_size=num_samples, num_values=3, min_prob=0.05)
    # model._read_information_DAG()
    # model._read_bayesian_model()
    data = model.generate_data()
    print(data.head())