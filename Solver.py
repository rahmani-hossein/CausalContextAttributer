import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import lasso_path
from typing import Tuple
from abc import ABC, abstractmethod
from sklearn.linear_model import LassoCV, Lasso
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from lightgbm import LGBMRegressor

class BASE_AME_Solver(ABC):
    """
    A base solver class.

    Methods:
        fit(self, masks: NDArray, outputs: NDArray, num_output_tokens: int) -> Tuple[NDArray, NDArray]:
            Fit the solver to the given data.
    """

    @abstractmethod
    def fit(self, X, y): ...



class LassoSolver(BASE_AME_Solver):
    """
    A LASSO solver using the scikit-learn library for estimating AME.

    Attributes:
        lasso_alpha (float):
            The alpha parameter for the LASSO regression. Defaults to 0.01.

    Methods:
        fit(self, masks: NDArray, outputs: NDArray, num_output_tokens: int) -> Tuple[NDArray, NDArray]:
            Fit the solver to the given data.
    """

    def __init__(self, coef_scaling) -> None:
        super().__init__()
        self.coef_scaling = coef_scaling

    def fit(self, X, y):
        """ 
        Do not use CrossValidation LAsso. It's bad. just use the lasso path like ElasticNet in R from the GLMNET package.
        """
        best_lambda, best_coef, best_intercept, alphas, coefs_unscaled, mses = lasso_path_in_sample_best(X, y)

        print("Optimal lambda:", best_lambda)
        return best_coef * np.sqrt(self.coef_scaling)


class SparseLassoSolver(BASE_AME_Solver):
    """
    A LASSO solver using the scikit-learn library for estimating AME.

    Attributes:
        lasso_alpha (float):
            The alpha parameter for the LASSO regression. Defaults to 0.01.

    Methods:
        fit(self, masks: NDArray, outputs: NDArray, num_output_tokens: int) -> Tuple[NDArray, NDArray]:
            Fit the solver to the given data.
    """

    def __init__(self, coef_scaling) -> None:
        super().__init__()
        self.coef_scaling = coef_scaling

    def fit(self, X, y):
        """ 
        Lambda 1se result
        """
        lasso_cv = LassoCV(cv=10, random_state=42).fit(X, y)

        # Compute lambda1se using our helper function
        lambda_1se = compute_lambda_1se(lasso_cv)
        print("Lambda that minimizes CV error (lambda_min):", lasso_cv.alpha_)
        print("Lambda 1se (sparser model within one SE of the best):", lambda_1se)

        # Optionally, refit a Lasso model with the lambda1se value:
        lasso_1se = Lasso(alpha=lambda_1se, random_state=42)
        lasso_1se.fit(X, y)

        return lasso_1se.coef_ * np.sqrt(self.coef_scaling)
    


def compute_lambda_1se(lasso_cv):
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
    min_std = std_mse[min_idx]
    
    # One-standard-error threshold
    threshold = min_mean + min_std
    
    # Find all alphas whose mean MSE is less than or equal to the threshold
    valid = mean_mse <= threshold
    
    # Choose the largest alpha among those (largest alpha gives the sparsest model)
    lambda_1se = alphas[valid].max()
    return lambda_1se






class LassoGLMNETSolver(BASE_AME_Solver):
    """
    A LASSO solver using the scikit-learn library for estimating AME.

    Attributes:
        lasso_alpha (float):
            The alpha parameter for the LASSO regression. Defaults to 0.01.

    Methods:
        fit(self, masks: NDArray, outputs: NDArray, num_output_tokens: int) -> Tuple[NDArray, NDArray]:
            Fit the solver to the given data.
    """

    def __init__(self, coef_scaling) -> None:
        super().__init__()
        self.coef_scaling = coef_scaling

    def fit(self, X, y):
        """ 
        Do not use CrossValidation LAsso. It's bad. just use the lasso path like ElasticNet in R from the GLMNET package.
        """
        best_lambda, best_coef, best_intercept, alphas, coefs_unscaled, mses = lasso_path_in_sample_best(X, y)

        print("Optimal lambda:", best_lambda)
        return best_coef * np.sqrt(self.coef_scaling)
    
class OrthogonalSolver(BASE_AME_Solver):
    def __init__(self, source_idx):
        super().__init__()
        self.propensity_model = GradientBoostingRegressor(max_depth=3, random_state=123)
        self.outcome_model = GradientBoostingRegressor(max_depth=3, random_state=123)
        self.final_model = GradientBoostingRegressor(max_depth=3, random_state=123)
        self.source_idx = source_idx

    
    def fit(self, X, y):
        """
        Estimate CATE for a single source
        
        Parameters:
        X (np.array): M x n binary matrix where M is number of samples, n is number of sources
        y (np.array): Target variable for each sample
        source_idx (int): Index of the source to estimate CATE for
        
        Returns:
        float: CATE estimate for the specified source
        """
        T = X[:, self.source_idx]
        X_reduced = np.delete(X, self.source_idx, axis=1)
        X_reduced = np.where(X_reduced> 0 , 1, 0)
        # Step 1: Estimate propensity score e(X) = P(T=1|X)
        self.propensity_model.fit(X_reduced, T)
        e_x = self.propensity_model.predict(X_reduced)
        
        # Step 2: Estimate outcome model m(X)
        self.outcome_model.fit(X_reduced, y)
        m_x = self.outcome_model.predict(X_reduced)
        
        # Step 3: Calculate pseudo-outcome
        pseudo_outcome = (y - m_x) / (T - e_x)
        


        # Step 4: Final regression for CATE
        w = (T - e_x) ** 2 
 
        # use a weighted regression ML model to predict the target with the weights.
        self.final_model.fit(X_reduced, pseudo_outcome, sample_weight=w)
        
        # Return average CATE
        return np.mean(self.final_model.predict(X_reduced))
    


from sklearn.preprocessing import StandardScaler

class StableOrthogonalSolver:
    def __init__(self, source_idx, epsilon=1e-8):
        """
        More stable Orthogonal Solver
        
        Parameters:
        source_idx (int): Index of the source to estimate effect for
        epsilon (float): Small constant to prevent division by zero
        """
        self.source_idx = source_idx
        self.epsilon = epsilon
        
        # More robust models
        self.propensity_model = GradientBoostingRegressor(
            max_depth=3, 
            random_state=123
        )
        self.outcome_model = GradientBoostingRegressor(
            max_depth=3, 
            random_state=123
        )
        self.final_model = GradientBoostingRegressor(
            max_depth=3, 
            random_state=123
        )
        
        # Scalers for numerical stability
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
    
    def fit(self, X, y):
        """
        More robust CATE estimation
        """
        # Prepare data
        T = X[:, self.source_idx]
        X_reduced = np.delete(X, self.source_idx, axis=1)
        X_reduced = np.where(X_reduced > 0, 1, 0)
        
        # Scale inputs for numerical stability
        X_scaled = self.X_scaler.fit_transform(X_reduced)
        y_scaled = self.y_scaler.fit_transform(y.reshape(-1, 1)).ravel()
        
        # Step 1: Estimate propensity score
        self.propensity_model.fit(X_scaled, T)
        e_x = self.propensity_model.predict(X_scaled)
        
        # Step 2: Estimate outcome model
        self.outcome_model.fit(X_scaled, y_scaled)
        m_x = self.outcome_model.predict(X_scaled)
        
        # Step 3: Robust pseudo-outcome calculation
        # Add small epsilon to prevent division by zero
        denominator = T - e_x + self.epsilon
        pseudo_outcome = (y_scaled - m_x) / denominator
        
        # Step 4: Weighted final regression
        # Use robust weighting to avoid numerical instability
        w = 1 / (np.abs(denominator) + self.epsilon)
        w = w / np.sum(w)  # Normalize weights
        
        self.final_model.fit(X_scaled, pseudo_outcome, sample_weight=w)
        
        # Estimate and inverse scale the result
        raw_cate = np.mean(self.final_model.predict(X_scaled))
        return raw_cate
    








def lasso_path_in_sample_best(
    X, y, 
    n_alphas=100, 
    eps=1e-3,
    tol=1e-8,
    max_iter=10000
):
    """
    Computes the Lasso path for a range of lambdas, standardizes features like glmnet,
    picks the alpha (lambda) with the lowest *in-sample* MSE, and returns
    the unscaled coefficients + intercept.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Response vector.
    n_alphas : int
        Number of alpha (lambda) values in the path.
    eps : float
        Ratio of alpha_min to alpha_max. 
        (alpha_max is where all coefficients go to zero).
    tol : float
        Tolerance for coordinate descent.
    max_iter : int
        Maximum number of iterations for the coordinate descent.

    Returns
    -------
    best_alpha : float
        Alpha (lambda) that yields minimum in-sample MSE.
    best_coef_ : ndarray of shape (n_features,)
        Coefficients unscaled (original data scale).
    best_intercept_ : float
        Intercept on original scale.
    alphas : ndarray of shape (n_alphas,)
        All alpha values used (descending).
    coefs_unscaled : ndarray of shape (n_features, n_alphas)
        Coefficients unscaled for each alpha in `alphas`.
    mses : ndarray of shape (n_alphas,)
        In-sample MSE for each alpha.
    """

    # -----------------------------
    # 1) Standardize X columns (like glmnet does by default),
    #    and center y so that we effectively handle intercept internally.
    # -----------------------------
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0, ddof=1)  # ddof=1 matches sample stdev in most stats packages
    # Avoid dividing by zero if any column has zero variance:
    X_std[X_std == 0] = 1.0

    Xs = (X - X_mean) / X_std  # standardized features

    y_mean = y.mean()
    ys = y - y_mean            # centered response

    # -----------------------------
    # 2) Compute Lasso path (no CV). This gives a series of alpha values
    #    from alpha_max down to alpha_max * eps, and coefficients in standardized space.
    # -----------------------------
    alphas, coefs, _ = lasso_path(
        Xs,
        ys,
        n_alphas=n_alphas,
        eps=eps,
        max_iter=max_iter,
        tol=tol
    )
    # coefs has shape (n_features, n_alphas)

    # -----------------------------
    # 3) For each alpha, unscale coefficients back to original X scale,
    #    compute intercept, evaluate in-sample MSE on original scale.
    # -----------------------------
    n_alphas_actual = len(alphas)
    mses = np.empty(n_alphas_actual, dtype=float)
    coefs_unscaled = np.empty_like(coefs)

    for i in range(n_alphas_actual):
        # Current standardized coefficients (p, )
        coef_std = coefs[:, i]

        # Unscale to original features: 
        #   beta_original_j = beta_std_j / X_std_j
        coef_orig = coef_std / X_std

        # Intercept in original scale:
        #   intercept = y_mean - sum_j( coef_orig_j * X_mean_j )
        intercept_orig = y_mean - np.sum(coef_orig * X_mean)

        coefs_unscaled[:, i] = coef_orig

        # Predictions on original scale:
        y_pred = X.dot(coef_orig) + intercept_orig

        # MSE on training data (original scale)
        mses[i] = np.mean((y - y_pred)**2)

    # -----------------------------
    # 4) Pick the alpha with the smallest in-sample MSE
    # -----------------------------
    best_idx = np.argmin(mses)
    best_alpha = alphas[best_idx]
    best_coef_ = coefs_unscaled[:, best_idx]
    # Recompute best intercept (could also just intercept_orig from loop above)
    best_intercept_ = y_mean - np.sum(best_coef_ * X_mean)
    
    return best_alpha, best_coef_, best_intercept_, alphas, coefs_unscaled, mses