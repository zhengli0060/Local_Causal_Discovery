'''
Tests the null hypothesis that X is independent from Y given S.
'''

from numpy.linalg import LinAlgError
import numpy as np
from scipy.stats import norm
from typing import Union, Set, Tuple, Dict, List
import pandas as pd
from pgmpy.estimators.CITests import chi_square,g_sq
import networkx as nx

def FisherZ_Test(X: Union[int, str], Y: Union[int, str], S: List[Union[int, str]], suffStat: Union[dict, pd.DataFrame], alpha: float = 0.05) -> Tuple[bool, float]:
    """
    Source code from the R package pcalg. 

    gaussCItest <- function(x,y,S,suffStat) 
    Perform Gaussian Conditional Independence Test.
    
    Parameters:
    X, Y: int or str - Variables to test for conditional independence.
    S: list - Conditioning set.
    suffStat: dict or pd.DataFrame - Sufficient statistics containing:
        If dict:
            'C': Correlation matrix.
            'n': Sample size.
        If pd.DataFrame:
            DataFrame containing the data to compute correlation matrix and sample size.
    
    Returns:
    bool - CI is True means independent, False means dependent.
    float - p-value of the test.
    """
    if isinstance(suffStat, pd.DataFrame):
        data = suffStat
        suffStat = {
            'C': data.corr().to_numpy(),
            'n': len(data)
        }
        col_index = {col: idx for idx, col in enumerate(data.columns)}
        X = col_index[X] if isinstance(X, str) else X
        Y = col_index[Y] if isinstance(Y, str) else Y
        S = [col_index[s] if isinstance(s, str) else s for s in S]

    def zStat(X: int, Y: int, S: list, C: np.ndarray, n: int) -> float:
        """
        Calculate Fisher's z-transform statistic of partial correlation.
        
        Parameters:
        X, Y: int - Variables to test for conditional independence.
        S: list - Conditioning set.
        C: np.ndarray - Correlation matrix.
        n: int - Sample size.
        
        Returns:
        float - z-statistic.
        """
        r = pcorOrder(X, Y, S, C)
        if r is None:
            return 0
        return np.sqrt(n - len(S) - 3) * 0.5 * np.log((1 + r) / (1 - r))

    def pcorOrder(i: int, j: int, k: list, C: np.ndarray, cut_at: float = 0.9999999) -> float:
        """
        Compute partial correlation.
        
        Parameters:
        i, j: int - Variables to compute partial correlation.
        k: list - Conditioning set.
        C: np.ndarray - Correlation matrix.
        
        Returns:
        float - Partial correlation coefficient.
        """
        if len(k) == 0:
            r = C[i, j]
        elif len(k) == 1:
            r = (C[i, j] - C[i, k[0]] * C[j, k[0]]) / np.sqrt((1 - C[j, k[0]]**2) * (1 - C[i, k[0]]**2))
        else:
            try:
                sub_matrix = C[np.ix_([i, j] + k, [i, j] + k)]
                PM = np.linalg.pinv(sub_matrix)
                r = -PM[0, 1] / np.sqrt(PM[0, 0] * PM[1, 1])
            except LinAlgError:
                return None
        if np.isnan(r):
            return 0
        return min(cut_at, max(-cut_at, r))

    z = zStat(X, Y, S, suffStat['C'], suffStat['n'])
    p_value = 2 * norm.cdf(-abs(z))
    CI = p_value > alpha  
    return CI, p_value



def G_sq_test(X: Union[int, str], Y: Union[int, str], S: List[Union[int, str]], data: pd.DataFrame, alpha: float = 0.05) -> Tuple[bool, float]:

    """
    Perform G-squared test for conditional independence.

    Parameters:
    X, Y: int or str - Variables to test for conditional independence.
    S: list - Conditioning set.
    data: pd.DataFrame - DataFrame containing the data to compute G-squared statistic.
    alpha: float - Significance level.

    Returns:
    bool - CI is True means independent, False means dependent.
    float - p-value of the test.
    """

    # _,p_value,_ = chi_square(X, Y, S, data, boolean=False)   # chi_square
    _,p_value,_ = g_sq(X, Y, S, data, boolean=False)   # g_sq
    CI = p_value > alpha
    return CI, p_value


def d_sep(X: Union[int, str], Y: Union[int, str], S: List[Union[int, str]], DAG: Union[np.ndarray, pd.DataFrame]) -> bool:
    """
    Check if X is d-separated from Y given S.

    Parameters:
    X, Y: int or str - Variables to test for conditional independence.
    S: list - Conditioning set.
    DAG: np.ndarray or pd.DataFrame - The DAG adjacency matrix. [num nodes, num nodes]

    """
    
    if not isinstance(DAG, (np.ndarray, pd.DataFrame)):
        raise TypeError("DAG must be a numpy array or pandas DataFrame.")
    if DAG.shape[0] != DAG.shape[1]:
        raise ValueError("DAG must be a square matrix.")
    if isinstance(DAG, pd.DataFrame):
        node_list = DAG.index.tolist()  # get the node list from the index of the DataFrame
        # create the nx.DiGraph object base on the above DAG adjacency matrix and node list
        G = nx.DiGraph()
        G.add_nodes_from(node_list)  
        for source in node_list:
            for target in node_list:
                if DAG.loc[source, target] == 1:
                    G.add_edge(source, target)
    else:
        G = nx.DiGraph(DAG)
    
    # old version: nx.d_separated(G, X, Y, set(S))
    return nx.is_d_separator(G, X, Y, set(S))



if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from Models.Continuous_model import Continuous_Model
    from Models.Discrete_model import Discrete_Model
    """
    Example usage:
        V1 -> V2 -> V3
    """
    num_nodes = 3
    DAG = np.array([[0, 1, 0],
                    [0, 0, 1],    
                    [0, 0, 0]])
    # Convert the adjacency matrix into a pandas DataFrame and assign indices to its rows and columns, labeled as V1, V2, V3, ..., Vn
    DAG = pd.DataFrame(DAG, index=[f'V{i+1}' for i in range(num_nodes)], columns=[f'V{i+1}' for i in range(num_nodes)])



    num_samples = 100
    model = Continuous_Model(DAG, function_type='linear', noise_type='gaussian', sample_size=num_samples)
    model._read_information_DAG()
    continuous_data = model.simulate_scm()

    model = Discrete_Model(DAG, sample_size=num_samples, num_values=3, min_prob=0.05)
    # model._read_information_DAG()
    # model._read_bayesian_model()
    discrete_data = model.generate_data()


    alpha = 0.05
    
    X = 'V1'
    Y = 'V3'
    S = ['V2']
    FisherZ_result = FisherZ_Test(X, Y, S, continuous_data)
    print(f"Fisher Z Test: CI={FisherZ_result[0]}, p-value={FisherZ_result[1]}")

    G_sq_result = G_sq_test(X, Y, S, discrete_data)
    print(f"G-squared Test: CI={G_sq_result[0]}, p-value={G_sq_result[1]}")

    d_sep_result = d_sep(X, Y, S, DAG)
    print(f"D-separation: {d_sep_result}")