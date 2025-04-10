import pandas as pd
import math
from typing import Union, Set, Tuple, Dict, List
from Conditional_Independence_Test.CI_test import FisherZ_Test,G_sq_test,d_sep,ci_test

"""
This file contains the implementation of the MB discovery algorithm.
The MB discovery algorithm is used to find the Markov blanket of a given variable in a network.
"""

def grow_shrink_mb(data: pd.DataFrame, target: Union[int, str], alpha: float=0.05, method_type: str = 'FisherZ') -> list:
    """
    Grow-Shrink (GS) Algorithm for Markov Blanket Discovery.
    
    Parameters:
        data (pd.DataFrame): The dataset as a pandas DataFrame.
        target (str): The target variable.
        alpha (float): Significance level for independence tests.
        method_type (str): Method type for CI test. Options are 'FisherZ', 'G_sq', or 'D_sep'.

    Returns:
        list: The Markov blanket of the target variable.

    References:
        - Margaritis D, Thrun S. Bayesian network induction via local neighborhoods[J]. Advances in neural information processing systems, 1999, 12.
    """

    MB = []  # Initialize the candidate Markov blanket as empty
    ntest = 0  # Initialize the number of tests performed
    Candidate = list(data.columns.difference([target]))  # Initialize the candidate set of variables

    # Growing phase
    Flag = True
    while Flag:
        Flag = False
        for Y in Candidate:
            ntest += 1
            if not ci_test(target, Y, list(MB),data, method_type, alpha):
                MB.add(Y)
                Candidate.remove(Y)
                Flag = True
                break

    # Shrinking phase
    Flag = True
    while Flag:
        Flag = False
        MB_temp = MB.copy()
        for Y in MB_temp:
            ntest += 1
            if ci_test(target, Y, list(MB.difference([Y])), data, method_type, alpha):
                MB.remove(Y)
                Flag = True
                break

    return MB, ntest

def TC_mb(data: pd.DataFrame, target: Union[int, str], alpha: float=0.05, method_type: str = 'FisherZ') -> list:
    """
    TC Algorithm for Markov Blanket Discovery.

    Parameters:
        data (pd.DataFrame): The dataset as a pandas DataFrame.
        target (str): The target variable.
        alpha (float): Significance level for independence tests.
        method_type (str): Method type for CI test. Options are 'FisherZ', 'G_sq', or 'D_sep'.

    Returns:
        list: The Markov blanket of the target variable.
    
    References:
        - Pellet J P, Elisseeff A. Using markov blankets for causal structure learning[J]. Journal of Machine Learning Research, 2008, 9(7).
    """
    sample_size = data.shape[0]  # Get the number of samples
    do_n = sample_size / 10
    ceil_result = math.floor(do_n)
    if ceil_result > 0:
        alpha = alpha/(ceil_result*10)
    MB = []  # Initialize the candidate Markov blanket as empty
    ntest = 0  # Initialize the number of tests performed
    Candidate = list(data.columns.difference([target]))  # Initialize the candidate set of variables

    for Y in Candidate:
        ntest += 1
        """
        Total conditioning
        """
        S = list(Candidate.copy()).remove(Y)  # Create a copy of MB and remove Y from it
        if not ci_test(target, Y, S, data, method_type, alpha):
            MB.append(Y)

    return MB, ntest


from Models.Continuous_model import simulate_dag
if __name__ == "__main__":
    # Example usage
    num_nodes = 50
    num_edges = 120
    DAG = simulate_dag(num_nodes, num_edges, 'ER') 
    DAG = pd.DataFrame(DAG, index=[f'V{i+1}' for i in range(num_nodes)], columns=[f'V{i+1}' for i in range(num_nodes)])
    target = 'V5'
    alpha = 0.05
    method_type = 'D_sep'
    mb_gs, ntest_gs = grow_shrink_mb(DAG, target, alpha, method_type)
    print(f"Markov Blanket (Grow-Shrink): {mb_gs}, Number of tests: {ntest_gs}")

    mb_tc, ntest_tc = TC_mb(DAG, target, alpha, method_type)
    print(f"Markov Blanket (TC): {mb_tc}, Number of tests: {ntest_tc}")