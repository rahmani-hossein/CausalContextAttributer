import numpy as np
def calculate_theoretical_bound(probs=[0.2, 0.4, 0.6, 0.8], n_samples=10000):
    """
    Calculate theoretical upper bound on prediction accuracy
    given the data generation process
    """
    # Generate many samples to simulate the process
    max_accuracies = []
    
    for _ in range(100):  # Run multiple trials
        # Simulate feature generation
        features = np.random.choice(probs, size=(n_samples, 8))
        binary_features = np.random.binomial(n=1, p=features)
        
        # Calculate maximum possible accuracy
        # This is based on knowing the true probability for each sample
        treatment = binary_features[:, 0]  # Assuming first column is treatment
        
        # Calculate Bayes optimal prediction
        true_probs = features[:, 0]
        optimal_predictions = (true_probs >= 0.5).astype(int)
        
        # Calculate maximum possible accuracy
        max_accuracy = np.mean(optimal_predictions == treatment)
        max_accuracies.append(max_accuracy)
    
    return {
        'mean_max_accuracy': np.mean(max_accuracies),
        'std_max_accuracy': np.std(max_accuracies),
        'theoretical_error_lower_bound': 1 - np.mean(max_accuracies)
    }

def theoretical_propensity_score(X):
    """
    Estimate the theoretical propensity score for each sample.
    X: numpy array of shape (n_samples, n_features)
    Assumes the first column is the treatment, the rest are features.
    """
    # Exclude the treatment column
    X_features = X[:, 1:]
    # Estimate p for each sample as the mean of the other features
    p_hat = X_features.mean(axis=1)
    return p_hat

def test_theoretical_propensity_score(probs=[0.2, 0.4, 0.6, 0.8], n_samples=10000, n_features=8, n_trials=100):
    """
    Generate X multiple times, compute theoretical propensity scores, and report statistics.
    """
    all_means = []
    all_stds = []
    for _ in range(n_trials):
        # For each sample, pick a p from probs
        ps = np.random.choice(probs, size=(n_samples, 1))
        # Generate all features (including treatment) as Bernoulli(p)
        X = np.random.binomial(n=1, p=ps, size=(n_samples, n_features))
        # Compute theoretical propensity scores (mean of features excluding treatment)
        p_hat = theoretical_propensity_score(X)
        all_means.append(np.mean(p_hat))
        all_stds.append(np.std(p_hat))
    print("\nTheoretical Propensity Score Estimation over Multiple Trials:")
    print(f"Mean of estimated propensity scores (averaged over trials): {np.mean(all_means):.3f} ± {np.std(all_means):.3f}")
    print(f"Std of estimated propensity scores (averaged over trials): {np.mean(all_stds):.3f} ± {np.std(all_stds):.3f}")
    return all_means, all_stds

# Run the test
if __name__ == "__main__":
    bounds = calculate_theoretical_bound()
    print(f"Theoretical Results:")
    print(f"Maximum Possible Accuracy: {bounds['mean_max_accuracy']:.3f} ± {bounds['std_max_accuracy']:.3f}")
    print(f"Minimum Possible Error Rate: {bounds['theoretical_error_lower_bound']:.3f}")
    # Test theoretical propensity score estimation
    test_theoretical_propensity_score()