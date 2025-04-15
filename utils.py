
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import lasso_path
from typing import Tuple
from abc import ABC, abstractmethod
from sklearn.linear_model import LassoCV, Lasso
from sklearn.linear_model import LogisticRegression, LinearRegression
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split



def generate_bernoulli_matrix(n_features, n_samples=1000, p_min=0.2, p_max=0.8):
        """
        Generate a matrix of Bernoulli trials where each row uses a probability sampled from uniform distribution.
        
        Args:
            n_features (int): Number of features (columns) in the matrix
            n_samples (int): Number of samples (rows) in the matrix, default 1000
            p_min (float): Minimum probability for uniform distribution, default 0.2
            p_max (float): Maximum probability for uniform distribution, default 0.8
            
        Returns:
            np.ndarray: Matrix of shape (n_samples, n_features) containing Bernoulli trials
            np.ndarray: Array of sampled probabilities used for each row
        """
    # Sample probabilities from uniform distribution
        sampled_probs = np.random.uniform(p_min, p_max, size=n_samples)
        
        # Initialize the matrix
        X = np.zeros((n_samples, n_features), dtype=float)
        
        # Generate Bernoulli trials for each row
        for i, p in enumerate(sampled_probs):
            X[i] = np.random.binomial(n=1, p=p, size=n_features)
        
        return X

def compute_lambda_1se(lasso_cv, cv = 10):
        """
        Compute lambda1se (largest alpha within one std error of the minimal CV error)
        from a fitted LassoCV model.
        
        Parameters:
            lasso_cv: A fitted instance of LassoCV.
            
        Returns:
            lambda_1se: The lambda value (alpha) that is within one standard error of the best.
        """
        alphas = lasso_cv.alphas_
        mse_path = lasso_cv.mse_path_  # shape: (n_alphas, n_folds)

        
        # Compute the mean and std of the MSE for each alpha
        mean_mse = mse_path.mean(axis=1)
        std_mse = mse_path.std(axis=1)
        
        # Find the minimum mean MSE and its index
        min_idx = mean_mse.argmin()
        min_mean = mean_mse[min_idx]
        min_std = std_mse[min_idx] / np.sqrt(cv)
        
        # One-standard-error threshold
        threshold = min_mean + min_std
        
        # Find all alphas whose mean MSE is less than or equal to the threshold
        valid = mean_mse <= threshold
        
        # Choose the largest alpha among those (largest alpha gives the sparsest model)
        lambda_1se = alphas[valid].max()
        return lambda_1se