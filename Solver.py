import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import lasso_path
from typing import Tuple
from abc import ABC, abstractmethod
from sklearn.linear_model import LassoCV, Lasso
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')
import utils
from econml.dml import NonParamDML


class BASE_AME_Solver(ABC):
    """
    A base solver class.

    Methods:
        fit(self, masks: NDArray, outputs: NDArray, num_output_tokens: int) -> Tuple[NDArray, NDArray]:
            Fit the solver to the given data.
    """

    @abstractmethod
    def fit(self, masks: NDArray, outputs: NDArray, num_output_tokens: int) -> Tuple[NDArray, NDArray]:
        ...



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

    def __init__(self, coef_scaling, num_folds = 10) -> None:
        super().__init__()
        self.coef_scaling = coef_scaling
        self.num_folds = num_folds

    def fit(self, X, y):
        """ 
        Lambda 1se result
        """
        lasso_cv = LassoCV(cv=self.num_folds, random_state=42).fit(X, y)

        # Compute lambda1se using our helper function
        lambda_1se = utils.compute_lambda_1se(lasso_cv, cv = self.num_folds)
        print("Lambda that minimizes CV error (lambda_min):", lasso_cv.alpha_)
        print("Lambda 1se (sparser model within one SE of the best):", lambda_1se)

        # Optionally, refit a Lasso model with the lambda1se value:
        lasso_1se = Lasso(alpha=lambda_1se, random_state=42)
        lasso_1se.fit(X, y)

        return lasso_1se.coef_ * np.sqrt(self.coef_scaling)
    

class CATENet(nn.Module):
    def __init__(self, input_dim):
        super(CATENet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32,1)
        )
    
    def forward(self, x):
        return self.network(x)

class PytorchRLearner:
    def __init__(self, 
                 learning_rate=0.01, 
                 n_epochs=2000,
                 batch_size=32,
                 random_state=123):
        """
        Initialize R-learner with PyTorch optimization
        """
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        
        # Initialize nuisance parameter models
        self.propensity_model = GradientBoostingClassifier(
            max_depth=3, 
            random_state=random_state
        )
        self.outcome_model = GradientBoostingRegressor(
            max_depth=3, 
            random_state=random_state
        )
        
    def _to_torch(self, x):
        """Convert numpy array to torch tensor"""
        # return torch.FloatTensor(x)
        return torch.FloatTensor(x.astype(np.float32))
    
    def create_dataloader(self, X, pseudo_outcomes):
        """Create PyTorch DataLoader"""
        X_torch = self._to_torch(X)
        pseudo_torch = self._to_torch(pseudo_outcomes)
        
        dataset = torch.utils.data.TensorDataset(X_torch, pseudo_torch)
        return torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
    
    
    
    def estimate_cate(self, X, T, y):
        """
        Estimate CATE using R-learning with PyTorch optimization
        """
        T = T.astype(int)
        
        # Split data for nuisance parameters
        X_train_nuisance, X_other_half, T_train_nuisance, T_other_half, \
        y_train_nuisance, y_other_half = train_test_split(
            X, T, y, test_size=0.5, random_state=self.random_state
        )
        
        # Step 1: Estimate nuisance parameters
        # Propensity score
        self.propensity_model.fit(X_train_nuisance, T_train_nuisance)
        e_x = self.propensity_model.predict_proba(X_other_half)[:, 1]
        
        # Outcome model
        self.outcome_model.fit(X_train_nuisance, y_train_nuisance)
        m_x = self.outcome_model.predict(X_other_half)
        
        # Step 2: Calculate pseudo-outcomes
        residual_y = y_other_half - m_x
        residual_t = T_other_half - e_x
        
        # Initialize CATE model
        self.cate_model = CATENet(X.shape[1])
        optimizer = optim.Adam(self.cate_model.parameters(), lr=self.learning_rate)
        
        # Convert to PyTorch tensors
        X_torch = self._to_torch(X_other_half)
        residual_y_torch = self._to_torch(residual_y.reshape(-1, 1))
        residual_t_torch = self._to_torch(residual_t.reshape(-1, 1))
        
        # Training loop
        for epoch in range(self.n_epochs):
            optimizer.zero_grad()
            
            # Forward pass
            tau_pred = self.cate_model(X_torch)
            
            # R-learner loss
            loss = torch.mean(
                (residual_y_torch - residual_t_torch * tau_pred)**2
            )
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 1000 == 0:
                print(f'Epoch [{epoch+1}/{self.n_epochs}], Loss: {loss.item():.4f}')

        self.cate_model.eval()
        with torch.no_grad():
            # uniform_samples = torch.from_numpy(utils.generate_bernoulli_matrix(n_features=X.shape[1], n_samples=1000, p_min=0, p_max=1))
            uniform_samples = utils.generate_bernoulli_matrix(n_features=X.shape[1], n_samples=1000, p_min=0, p_max=1)
            uniform_samples = torch.FloatTensor(uniform_samples.astype(np.float32))
            cate_estimates = self.cate_model(uniform_samples).cpu().numpy().flatten()
        
        return {
            'cate_estimates': cate_estimates,
            'ate': np.mean(cate_estimates),
            'final_loss': loss.item()
        }

def estimate_all_treatments(X, y):
    """
    Estimate CATE for each column in X sequentially as treatment
    Also compare with EconML NonParamDML for each treatment.
    """
    n_treatments = X.shape[1]
    results = {}
    econml_results = {}
    for i in range(n_treatments):
        print(f"\nEstimating CATE for treatment {i} (PyTorch R-learner)")
        # Extract current treatment and remaining features
        T = X[:, i]
        X_reduced = np.delete(X, i, axis=1)
        # PyTorch R-learner
        rlearner = PytorchRLearner(
            learning_rate=0.01,
            n_epochs=2000,
            batch_size=32
        )
        print(f'Treatment feature {i} shape is {T.shape} for MYorthogonal method.')
        treatment_results = rlearner.estimate_cate(X_reduced, T, y)

        results[f'treatment_{i}'] = {
            'ate': treatment_results['ate'],
            'cate_estimates': treatment_results['cate_estimates'],
            'final_loss': treatment_results['final_loss']
        }
        # EconML DML
        print(f"Estimating CATE for treatment {i} (EconML NonParamDML)")
        econml_learner = EconMLDMLLearner()
        print(f'X_reduced shape is {X_reduced.shape}')
        print(f'Treatmentfeature {i} shape is {T.shape}')

        econml_treatment_results = econml_learner.estimate_cate( X=X_reduced, T=T, y=y)
        econml_results[f'treatment_{i}'] = {
            'ate': econml_treatment_results['ate'],
            'cate_estimates': econml_treatment_results['cate_estimates'],
            'final_loss': econml_treatment_results['final_loss']
        }
        # Print comparison
        print(f"PyTorch R-learner ATE: {treatment_results['ate']:.3f}")
        print(f"EconML DML ATE: {econml_treatment_results['ate']:.3f}")
        print(f"PyTorch R-learner CATE std: {np.std(treatment_results['cate_estimates']):.3f}")
        print(f"EconML DML CATE std: {np.std(econml_treatment_results['cate_estimates']):.3f}")
        print(f"PyTorch R-learner Final Loss: {treatment_results['final_loss']}")
        print(f"EconML DML Final Loss: {econml_treatment_results['final_loss']}")
    return {'pytorch': results, 'econml': econml_results}

# # Example usage:
# if __name__ == "__main__":
#     # Load data
#     X = np.load("data/X_M_max.npy")
#     y = np.load("data/y_M_max.npy")
    
#     # Get results for all treatments
#     all_results = estimate_all_treatments(X, y)
    
#     # Print summary
#     print("\nResults Summary:")
#     print("-" * 50)
#     for treatment, results in all_results.items():
#         print(f"\n{treatment}:")
#         print(f"ATE: {results['ate']:.3f}")
#         print(f"Final Loss: {results['final_loss']:.4f}")
#         print(f"CATE std: {np.std(results['cate_estimates']):.3f}")

class EconMLDMLLearner:
    def __init__(self, n_estimators=100, max_depth=3, random_state=123):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def estimate_cate(self, X, T, y):
        n_train = X.shape[0]
        # Use GradientBoostingRegressor for all models as in your example
        T = (T > 0.5).astype(np.int32)  # Convert to binary (0 or 1)
        print(f"T shape: {T.shape}, Unique values: {np.unique(T)}")  # Debug print
        model_t = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=max(1, int(n_train/100)),
            random_state=self.random_state
        )
        model_y = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=max(1, int(n_train/100)),
            random_state=self.random_state
        )
        model_final = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=max(1, int(n_train/100)),
            random_state=self.random_state
        )
        dml = NonParamDML(
            model_t=model_t,
            model_y=model_y,
            model_final=model_final,
            discrete_treatment=True,
            random_state=self.random_state
        )
        dml.fit(y, T, X=X, W=None)
        # For comparison, generate CATE estimates on a uniform sample as in PytorchRLearner
        uniform_samples = utils.generate_bernoulli_matrix(n_features=X.shape[1], n_samples=1000, p_min=0, p_max=1)
        cate_estimates = dml.effect(uniform_samples)
        return {
            'cate_estimates': cate_estimates,
            'ate': np.mean(cate_estimates),
            'final_loss': None  # Not directly available from econml
        }

def estimate_all_treatments_econml(X, y):
    """
    Estimate CATE for each column in X sequentially as treatment using EconML NonParamDML
    """
    n_treatments = X.shape[1]
    results = {}
    for i in range(n_treatments):
        print(f"\n[EconML] Estimating CATE for treatment {i}")
        T = X[:, i]
        X_reduced = np.delete(X, i, axis=1)
        learner = EconMLDMLLearner()
        treatment_results = learner.estimate_cate(X_reduced, T, y)
        results[f'treatment_{i}'] = {
            'ate': treatment_results['ate'],
            'cate_estimates': treatment_results['cate_estimates'],
            'final_loss': treatment_results['final_loss']
        }
    return results