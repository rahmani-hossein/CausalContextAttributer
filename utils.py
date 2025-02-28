
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