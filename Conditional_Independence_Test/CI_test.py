'''
Tests the null hypothesis that X is independent from Y given S.
'''

from numpy.linalg import LinAlgError
import numpy as np
from scipy.stats import norm
from typing import Union, Set, Tuple, Dict, List
import pandas as pd
from pgmpy.estimators.CITests import chi_square,g_sq

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

    #_,p_value,_ = chi_square(X, Y, S, data, boolean=False)   # g_sq
    _,p_value,_ = g_sq(X, Y, S, data, boolean=False)   # g_sq
    CI = p_value > alpha
    return CI, p_value




if __name__ == "__main__":
    # Example usage
    data = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [5, 4, 3, 2, 1],
        'C': [1, 1, 1, 0, 0]
    })
    X = 'A'
    Y = 'B'
    S = ['C']
    alpha = 0.05

    CI_result = FisherZ_Test(X, Y, S, data)
    print(f"Fisher Z Test: CI={CI_result[0]}, p-value={CI_result[1]}")

    G_sq_result = G_sq_test(X, Y, S, data)
    print(f"G-squared Test: CI={G_sq_result[0]}, p-value={G_sq_result[1]}")