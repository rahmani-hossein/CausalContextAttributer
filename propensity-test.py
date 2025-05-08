import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')
def estimate_propensity_scores(X, treatment):
    """
    Estimate propensity scores using logistic regression.
    
    Parameters:
    -----------
    X : numpy array
        Covariate matrix where each column is a binary feature
    treatment : numpy array
        Treatment indicator (will be converted to binary)
        
    Returns:
    --------
    dict containing:
        - model: fitted logistic regression model
        - propensity_scores: estimated propensity scores
        - performance_metrics: dictionary of model performance metrics
    """
    # Convert treatment to binary
    # First, let's print some information about the treatment variable
    print("Treatment variable statistics before conversion:")
    print(f"Unique values: {np.unique(treatment)}")
    print(f"Min: {np.min(treatment)}, Max: {np.max(treatment)}")
    
    # Convert to binary using threshold
    if not np.array_equal(treatment, treatment.astype(bool)):
        print("Converting treatment to binary...")
        # If the data is continuous, we'll use 0.5 as a threshold
        treatment_binary = (treatment >= 0.5).astype(int)
        print(f"Unique values after conversion: {np.unique(treatment_binary)}")
    else:
        treatment_binary = treatment.astype(int)
    
    # Ensure X is numeric
    X = X.astype(float)
    
    # Split data for validation
    X_train, X_test, t_train, t_test = train_test_split(
        X, treatment_binary, test_size=0.2, random_state=42
    )
    # Define models to try
    random_state = 42
    models = {
        'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            min_samples_leaf=5,
            random_state=random_state
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            min_samples_leaf=5,
            random_state=random_state
        ),
        'Neural Network': MLPClassifier(
            hidden_layer_sizes=(10, 5),
            max_iter=1000,
            random_state=random_state
        )
    }
    
    results = {}
    
    for name, model in models.items():
        # Fit model
        model.fit(X_train, t_train)
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Get cross-validation scores
        cv_scores = cross_val_score(model, X, treatment_binary, cv=5)
        
        # Store results
        results[name] = {
            'model': model,
            'classification_report': classification_report(t_test, y_pred),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        if hasattr(model, 'feature_importances_'):
            results[name]['feature_importances'] = model.feature_importances_
        elif hasattr(model, 'coef_'):
            results[name]['feature_importances'] = model.coef_[0]
    
    return results

# Load and prepare your data
X_full = np.load("CausalContextAttributer/data/X_M_max.npy")
y_full = np.load("CausalContextAttributer/data/y_M_max.npy")
m = X_full.shape[0]
X = X_full [0:m//2,:]
source_idx = 0
T = X[:, source_idx]
X_reduced = np.delete(X, source_idx, axis=1)
X_reduced = np.where(X_reduced> 0 , 1, 0)
# Estimate propensity scores
results = estimate_propensity_scores(X_reduced, T)

# Print results
for name, result in results.items():
    print(f"\n{'-'*50}")
    print(f"{name} Results:")
    print(f"{'-'*50}")
    print("\nClassification Report:")
    print(result['classification_report'])
    print(f"\nCross-validation score: {result['cv_mean']:.3f} (+/- {result['cv_std']*2:.3f})")
    
    if 'feature_importances' in result:
        print("\nFeature Importances:")
        for i, importance in enumerate(result['feature_importances']):
            print(f"Feature {i+1}: {importance:.3f}")