import pandas as pd
import math
from typing import Union, Set, Tuple, Dict, List
from Conditional_Independence_Test.CI_test import FisherZ_Test,G_sq_test,d_sep,ci_test

"""
This file contains the implementation of the MB discovery algorithm.
The MB discovery algorithm is used to find the Markov blanket of a given variable in a network.
"""

def grow_shrink_mb(data: pd.DataFrame, target: Union[int, str], alpha: float=0.05, method_type: str = 'FisherZ', latent_variables: list = None) -> list:
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

    """
    latent variables is not None means that we test the oracle MB in the presence of latent variables.
    """
    if latent_variables is not None :
        if data.shape[0] != data.shape[1]:
            raise ValueError("The latent variables are not None. The data must be a square matrix.")
        if method_type != 'D_sep':
            raise ValueError("The latent variables are not None. The method_type must be 'D_sep'.")


    MB = []  # Initialize the candidate Markov blanket as empty
    ntest = 0  # Initialize the number of tests performed
    if latent_variables is None:
        Candidate = list(data.columns.difference([target]))  # Initialize the candidate set of variables
    else:
        Candidate = list(data.columns.difference([target] + latent_variables))

    # Growing phase
    Flag = True
    while Flag:
        Flag = False
        for Y in Candidate:
            ntest += 1
            if not ci_test(target, Y, MB, data, method_type, alpha):
                MB.append(Y)
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
            S = [var for var in MB if var != Y]  # Create a copy of MB and remove Y from it
            if ci_test(target, Y, S, data, method_type, alpha):
                MB.remove(Y)
                Flag = True
                break

    return MB, ntest

def TC_mb(data: pd.DataFrame, target: Union[int, str], alpha: float=0.05, method_type: str = 'FisherZ', latent_variables: list = None) -> list:
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

    """
    latent variables is not None means that we test the oracle MB in the presence of latent variables.
    """
    if latent_variables is not None :
        if data.shape[0] != data.shape[1]:
            raise ValueError("The latent variables are not None. The data must be a square matrix.")
        if method_type != 'D_sep':
            raise ValueError("The latent variables are not None. The method_type must be 'D_sep'.")


    sample_size = data.shape[0]  # Get the number of samples
    do_n = sample_size / 10
    ceil_result = math.floor(do_n)
    if ceil_result > 0:
        alpha = alpha/(ceil_result*10)
    MB = []  # Initialize the candidate Markov blanket as empty
    ntest = 0  # Initialize the number of tests performed

    if latent_variables is None:
        Candidate = list(data.columns.difference([target]))  # Initialize the candidate set of variables
    else:
        Candidate = list(data.columns.difference([target] + latent_variables))

    for Y in Candidate:
        ntest += 1
        """
        Total conditioning
        """
        S = [var for var in Candidate if var != Y]  # Create a copy of MB and remove Y from it
        if not ci_test(target, Y, S, data, method_type, alpha):
            MB.append(Y)

    return MB, ntest

# num_samples = 1000
# model = Continuous_Model(DAG, function_type='linear', noise_type='gaussian', sample_size=num_samples)
# model._read_information_DAG()
# data = model.simulate_scm()
from Models.Continuous_model import simulate_dag, Continuous_Model
import random
if __name__ == "__main__":
    """
    Example 1: DAG without latent variables, using d-sep
    """
    # Parameters
    num_nodes = 50
    num_edges = 250
    alpha = 0.05
    method_type = 'D_sep'

    # Simulate a DAG
    DAG = simulate_dag(num_nodes, num_edges, 'ER')
    DAG = pd.DataFrame(DAG, index=[f'V{i+1}' for i in range(num_nodes)], columns=[f'V{i+1}' for i in range(num_nodes)])

    # Randomly select a target variable
    target = random.choice(DAG.columns)
    print(f"Target Variable: {target}")

    # Run the algorithms
    mb_gs, ntest_gs = grow_shrink_mb(DAG, target, alpha, method_type)
    mb_tc, ntest_tc = TC_mb(DAG, target, alpha, method_type)

    # Output results
    print(f"Markov Blanket (Grow-Shrink Algorithm): {mb_gs}")
    print(f"Number of Conditional Independence Tests (Grow-Shrink): {ntest_gs}")
    print(f"Markov Blanket (Total Conditioning Algorithm): {mb_tc}")
    print(f"Number of Conditional Independence Tests (Total Conditioning): {ntest_tc}")

    # Compare results
    if set(mb_gs) == set(mb_tc):
        print("The Markov Blankets obtained from the Grow-Shrink and Total Conditioning algorithms are identical.")
    else:
        print("The Markov Blankets obtained from the Grow-Shrink and Total Conditioning algorithms differ.")

    """
    Example 2: DAG with latent variables, using d-sep
    """
    # Parameters
    num_nodes = 50
    num_edges = 120
    alpha = 0.05
    method_type = 'D_sep'

    # Simulate a DAG
    DAG = simulate_dag(num_nodes, num_edges, 'ER')
    DAG = pd.DataFrame(DAG, index=[f'V{i+1}' for i in range(num_nodes)], columns=[f'V{i+1}' for i in range(num_nodes)])

    # Determine latent variables
    candidate_latent = [node for node in DAG.columns if DAG.loc[:, node].sum() >= 2]
    latent_num = int(len(candidate_latent) * 0.4)
    latent_variables = random.sample(candidate_latent, latent_num)
    print(f"Candidate Latent Variables: {candidate_latent}")
    print(f"Number of Latent Variables: {latent_num}, Latent Variables: {latent_variables}")

    # Randomly select a target variable excluding latent variables
    candidates = list(set(DAG.columns) - set(latent_variables))
    if not candidates:
        raise ValueError("No valid target variables available after excluding latent variables.")
    target = random.choice(candidates)
    print(f"Target Variable: {target}")

    # Run the algorithms
    mb_gs, ntest_gs = grow_shrink_mb(DAG, target, alpha, method_type, latent_variables=latent_variables)
    mb_tc, ntest_tc = TC_mb(DAG, target, alpha, method_type, latent_variables=latent_variables)

    # Output results
    print(f"Markov Blanket (Grow-Shrink Algorithm): {mb_gs}")
    print(f"Number of Conditional Independence Tests (Grow-Shrink): {ntest_gs}")
    print(f"Markov Blanket (Total Conditioning Algorithm): {mb_tc}")
    print(f"Number of Conditional Independence Tests (Total Conditioning): {ntest_tc}")

    # Compare results
    if set(mb_gs) == set(mb_tc):
        print("The Markov Blankets obtained from the Grow-Shrink and Total Conditioning algorithms are identical.")
    else:
        print("The Markov Blankets obtained from the Grow-Shrink and Total Conditioning algorithms differ.")
