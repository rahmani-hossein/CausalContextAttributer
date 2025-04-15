# attribution_comparison.py
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
# from tqdm.auto import tqdm
import time
import os
from dotenv import load_dotenv
from datasets import load_dataset
from scipy.stats import spearmanr
from tqdm import tqdm
import re
from llm_handler import LLM_Handler
from Attributer import Attributer


class AttributionBenchmark:
    def __init__(self,
        model_name: str = "meta-llama/Llama-3.2-1B",
        dataset_name: str = "ag_news",
        num_samples: int = 100,
        num_attribution_samples = 1000):
        """
        Initialize attribution comparison experiment.
        
        Args:
            model_name: Name of the LLM model to use
            dataset_name: Name of the dataset to use
            num_samples: Number of samples to evaluate between methods (AME, shap, Integrated gradients or others)
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.num_samples = num_samples
        self.num_attribution_samples = num_attribution_samples
        load_dotenv('/home/hrahmani/CausalContextAttributer/config.env')
        self._load_dataset()
        self.llm_handler = LLM_Handler(
            class_labels=self.class_labels,
            hf_auth_token=os.environ.get("HF_API_KEY"),
            model_name=self.model_name
        )
        self.attributer = Attributer(llm_handler=self.llm_handler)
    

    def _load_dataset(self):
        """Load and prepare the dataset."""
        dataset = load_dataset(self.dataset_name, split="test")
        indices = np.random.choice(len(dataset), self.num_samples, replace=False)
        self.data = [dataset[int(i)] for i in indices]
        # print(self.data)
        if self.dataset_name =="ag_news":
            self.class_labels = ["World", "Sports", "Business", "Sci/Tech"]
        self.index_to_label = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}
    

    def compute_spearman_correlation(
        shap_attrs: List[Tuple[str, float]],
        ame_attrs: List[Tuple[str, float]]
    ) -> float:
        """Compute Spearman correlation between SHAP and AME attributions."""
        # Create word to score mapping
        shap_scores = {word: score for word, score in shap_attrs}
        ame_scores = {word: score for word, score in ame_attrs}
        
        # Get common words
        common_words = set(shap_scores.keys()) & set(ame_scores.keys())
        
        # Create score arrays
        shap_array = [shap_scores[word] for word in common_words]
        ame_array = [ame_scores[word] for word in common_words]
        
        # Compute correlation
        correlation, _ = spearmanr(shap_array, ame_array)
        return correlation

    def compute_log_prob_drop(self, text: str, attributions: List[Tuple[str, float]], k: int = 1) -> Dict[str, float]:
        """
        Compute the drop in log probability when removing top k most important words.
        
        Args:
            text: Original input text
            attributions: List of (word, score) tuples from attribution method
            k: Number of top words to remove
            
        Returns:
            Dict containing metrics about the probability changes
        """
        # Get original prediction and probability
        original_label = self.llm_handler.get_predicted_class(text)
        original_metrics = self.llm_handler.get_classification_metrics(text, original_label)
        original_log_prob = original_metrics['normalized_log_prob_given_label']
        
        # Sort words by absolute attribution score
        sorted_words = sorted(attributions, key=lambda x: abs(x[1]), reverse=True)
        words_to_remove = [word for word, _ in sorted_words[:k]]
        
        # Create modified text by replacing top k words with __
        modified_text = text
        for word in words_to_remove:
            modified_text = re.sub(r'\b' + re.escape(word) + r'\b', '__', modified_text)
        
        # Get new prediction and probability
        modified_metrics = self.llm_handler.get_classification_metrics(modified_text, original_label)
        modified_log_prob = modified_metrics['normalized_log_prob_given_label']
        
        return original_log_prob - modified_log_prob
        # return {
        #     'original_label': original_label,
        #     'original_log_prob': original_log_prob,
        #     'modified_log_prob': modified_log_prob,
        #     'log_prob_drop': original_log_prob - modified_log_prob,
        #     'modified_label': modified_label,
        #     'removed_words': words_to_remove
        # }

    def run_comparison(self):
        for sample in tqdm(self.data,  desc="Processing samples"):
            original_text = sample['text']
            original_label = self.index_to_label[sample['label']]
            
            self.attributer.attribute(original_text, num_datasets=self.num_attribution_samples)

        
    def run_benchmark(self, methods: List[str] = ['shap', 'lasso'], k_values: List[int] = [1, 2, 3], **kwargs) -> Dict:
        """
        Run attribution benchmark comparing different methods.
        
        Args:
            methods: List of attribution methods to compare
            k_values: List of k values for top-k word removal
            **kwargs: Additional arguments passed to attribution methods
            
        Returns:
            Dict containing evaluation metrics for each method and k value
        """
        results = {}
        
        for method in methods:
            results[method] = {k: {
                'log_prob_drops': [],
                'label_changes': [],
                'times': []
            } for k in k_values}
            
            # Process each text in dataset
            for item in tqdm(self.data, desc=f"Evaluating {method}"):
                text = item['text']
                true_label = self.class_labels[item['label']]
                
                # Time the attribution computation
                start_time = time.time()
                if method == 'shap':
                    attributions = self.attributer.attribute_shap(text, true_label, **kwargs)
                else:  # lasso or ate
                    coeffs = self.attributer.attribute(text, method_name=method, **kwargs)
                    words = re.findall(r'\b\w+\b', text)
                    if isinstance(coeffs, (float, int)):
                        coeffs = [coeffs] * len(words)
                    attributions = list(zip(words, coeffs))
                attribution_time = time.time() - start_time
                
                # Compute metrics for each k
                for k in k_values:
                    metrics = self.compute_log_prob_drop(text, attributions, k)
                    
                    results[method][k]['log_prob_drops'].append(metrics['log_prob_drop'])
                    results[method][k]['label_changes'].append(
                        metrics['original_label'] != metrics['modified_label'])
                    results[method][k]['times'].append(attribution_time)
        
        # Compute summary statistics
        summary = {}
        for method in methods:
            summary[method] = {}
            for k in k_values:
                drops = results[method][k]['log_prob_drops']
                changes = results[method][k]['label_changes']
                times = results[method][k]['times']
                
                summary[method][k] = {
                    'mean_log_prob_drop': np.mean(drops),
                    'std_log_prob_drop': np.std(drops),
                    'label_change_rate': np.mean(changes),
                    'mean_attribution_time': np.mean(times)
                }
        
        return summary
    
    
    

    def run_benchmark(self):
        """Run the full benchmark comparison."""
        results = {
            'log_prob_drops': {
                'shap': {k: [] for k in self.k_values},
                'ame': {k: [] for k in self.k_values}
            },
            'correlations': [],
            'computation_times': {'shap': [], 'ame': []}
        }
        
        for sample in tqdm(self.dataset, desc="Processing samples"):
            text = sample['text']
            label = self.class_labels[sample['label']]  # Convert numeric label to string
            
            # Get SHAP attributions
            shap_attrs = self.attributer.attribute_shap(text, label, nsamples=100)
            
            # Get AME attributions (modified to return tuples)
            ame_coeffs = self.attributer.attribute(text, num_datasets=100)
            # Convert AME coefficients to (word, score) tuples
            words = text.split()
            ame_attrs = list(zip(words, ame_coeffs))
            
            # Compute correlation
            shap_scores = {word: score for word, score in shap_attrs}
            ame_scores = {word: score for word, score in ame_attrs}
            common_words = set(shap_scores.keys()) & set(ame_scores.keys())
            
            if common_words:
                shap_values = [shap_scores[w] for w in common_words]
                ame_values = [ame_scores[w] for w in common_words]
                correlation, _ = spearmanr(shap_values, ame_values)
                results['correlations'].append(correlation)
            
            # Compute log probability drops for different k
            for k in self.k_values:
                shap_drop = self.compute_log_prob_drop(text, shap_attrs, k, label)
                ame_drop = self.compute_log_prob_drop(text, ame_attrs, k, label)
                
                results['log_prob_drops']['shap'][k].append(shap_drop)
                results['log_prob_drops']['ame'][k].append(ame_drop)
        
        return results

    def plot_results(self, results: Dict):
        """Create plots similar to the paper figure."""
        # Plot log probability drops
        plt.figure(figsize=(12, 6))
        
        # Prepare data for plotting
        x_positions = np.arange(len(self.k_values))
        width = 0.35
        
        # Plot bars for each method
        for i, method in enumerate(['shap', 'ame']):
            means = [np.mean(results['log_prob_drops'][method][k]) for k in self.k_values]
            stds = [np.std(results['log_prob_drops'][method][k]) for k in self.k_values]
            
            plt.bar(x_positions + i*width, means, width, 
                   label=method.upper(),
                   yerr=stds, capsize=5)
        
        plt.xlabel('k value')
        plt.ylabel('Log Probability Drop')
        plt.title('Top-k Log Probability Drop Comparison')
        plt.xticks(x_positions + width/2, [f'k={k}' for k in self.k_values])
        plt.legend()
        plt.tight_layout()
        plt.savefig('log_prob_drops.png')
        plt.close()
        
        # Plot correlation distribution
        plt.figure(figsize=(8, 6))
        plt.hist(results['correlations'], bins=20)
        plt.xlabel('Spearman Correlation')
        plt.ylabel('Count')
        plt.title('Distribution of SHAP-AME Correlations')
        plt.savefig('correlations.png')
        plt.close()

def main():
    # Initialize benchmark
    benchmark = AGNewsBenchmark(
        num_samples=100,
        k_values=[1, 3, 5]
    )
    
    # Run benchmark
    print("Running benchmark...")
    results = benchmark.run_benchmark()
    
    # Plot results
    print("Creating plots...")
    benchmark.plot_results(results)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("Average Spearman correlation:", np.mean(results['correlations']))
    for k in benchmark.k_values:
        print(f"\nTop-{k} Log Probability Drop:")
        print(f"SHAP: {np.mean(results['log_prob_drops']['shap'][k]):.3f} ± {np.std(results['log_prob_drops']['shap'][k]):.3f}")
        print(f"AME:  {np.mean(results['log_prob_drops']['ame'][k]):.3f} ± {np.std(results['log_prob_drops']['ame'][k]):.3f}")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    attBenchmark = AttributionBenchmark(model_name="meta-llama/Llama-3.2-1B", dataset_name= "ag_news", num_samples=10)













        
    def compute_top_k_importance(self, attributions: List[Tuple[str, float]], k: int) -> List[str]:
        """Get top k most important words based on attribution scores."""
        sorted_attrs = sorted(attributions, key=lambda x: abs(x[1]), reverse=True)
        return [word for word, _ in sorted_attrs[:k]]
    
    def compute_word_drop_impact(
        self,
        text: str,
        top_k_words: List[str],
        original_label: str
    ) -> float:
        """Compute impact of removing top k words on classification probability."""
        # Remove top k words
        words = text.split()
        filtered_words = [w for w in words if w not in top_k_words]
        filtered_text = " ".join(filtered_words)
        
        # Get new classification probability
        metrics = self.llm_handler.get_classification_metrics(filtered_text, original_label)
        return metrics['normalized_prob_given_label']
    
    
    def run_comparison(self) -> Dict:
        """Run the full comparison experiment."""
        results = {
            'top_k_importance': {'shap': [], 'ame': []},
            'word_drop_impact': {'shap': [], 'ame': []},
            'computation_time': {'shap': [], 'ame': []},
            'spearman_correlation': [],
            'stability': {'shap': [], 'ame': []}
        }
        
        for sample in tqdm(self.data, desc="Processing samples"):
            text = sample['text']
            true_label = sample['label']
            
            # Get predicted label
            pred_label = self.llm_handler.get_predicted_class(text)
            
            # Compute SHAP attributions
            start_time = time.time()
            shap_attrs = self.attributer.attribute_shap(
                text, pred_label, nsamples=self.nsamples_shap
            )
            shap_time = time.time() - start_time
            
            # Compute AME attributions
            start_time = time.time()
            ame_attrs = self.attributer.attribute(text, num_datasets=100)
            ame_time = time.time() - start_time
            
            # Store computation times
            results['computation_time']['shap'].append(shap_time)
            results['computation_time']['ame'].append(ame_time)
            
            # Compute top-k importance
            for k in [3, 5, 10]:
                shap_top_k = self.compute_top_k_importance(shap_attrs, k)
                ame_top_k = self.compute_top_k_importance(ame_attrs, k)
                
                results['top_k_importance']['shap'].append(shap_top_k)
                results['top_k_importance']['ame'].append(ame_top_k)
                
                # Compute word drop impact
                shap_impact = self.compute_word_drop_impact(text, shap_top_k, pred_label)
                ame_impact = self.compute_word_drop_impact(text, ame_top_k, pred_label)
                
                results['word_drop_impact']['shap'].append(shap_impact)
                results['word_drop_impact']['ame'].append(ame_impact)
            
            # Compute Spearman correlation
            correlation = self.compute_spearman_correlation(shap_attrs, ame_attrs)
            results['spearman_correlation'].append(correlation)
            
            # Compute stability (run each method twice)
            shap_attrs2 = self.attributer.attribute_shap(
                text, pred_label, nsamples=self.nsamples_shap
            )
            ame_attrs2 = self.attributer.attribute(text, num_datasets=100)
            
            stability_shap = self.compute_spearman_correlation(shap_attrs, shap_attrs2)
            stability_ame = self.compute_spearman_correlation(ame_attrs, ame_attrs2)
            
            results['stability']['shap'].append(stability_shap)
            results['stability']['ame'].append(stability_ame)
        
        return results
    
    def visualize_results(self, results: Dict):
        """Create visualizations of the comparison results."""
        # 1. Computation Time Comparison
        plt.figure(figsize=(10, 6))
        plt.boxplot([
            results['computation_time']['shap'],
            results['computation_time']['ame']
        ], labels=['SHAP', 'AME'])
        plt.title('Computation Time Comparison')
        plt.ylabel('Time (seconds)')
        plt.savefig('computation_time.png')
        plt.close()
        
        # 2. Spearman Correlation Distribution
        plt.figure(figsize=(10, 6))
        plt.hist(results['spearman_correlation'], bins=20)
        plt.title('Distribution of Spearman Correlations')
        plt.xlabel('Correlation Coefficient')
        plt.ylabel('Count')
        plt.savefig('spearman_correlation.png')
        plt.close()
        
        # 3. Stability Comparison
        plt.figure(figsize=(10, 6))
        plt.boxplot([
            results['stability']['shap'],
            results['stability']['ame']
        ], labels=['SHAP', 'AME'])
        plt.title('Attribution Stability Comparison')
        plt.ylabel('Stability Score')
        plt.savefig('stability.png')
        plt.close()
        
        # 4. Word Drop Impact Comparison
        plt.figure(figsize=(10, 6))
        plt.boxplot([
            results['word_drop_impact']['shap'],
            results['word_drop_impact']['ame']
        ], labels=['SHAP', 'AME'])
        plt.title('Word Drop Impact Comparison')
        plt.ylabel('Classification Probability Drop')
        plt.savefig('word_drop_impact.png')
        plt.close()

def main():
    # Initialize experiment
    experiment = AttributionComparison(
        model_name="meta-llama/Llama-3.2-1B",
        num_samples=100,
        nsamples_shap=100
    )
    
    # Run comparison
    print("Running attribution comparison...")
    results = experiment.run_comparison()
    
    # Visualize results
    print("Creating visualizations...")
    experiment.visualize_results(results)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Average SHAP computation time: {np.mean(results['computation_time']['shap']):.2f}s")
    print(f"Average AME computation time: {np.mean(results['computation_time']['ame']):.2f}s")
    print(f"Average Spearman correlation: {np.mean(results['spearman_correlation']):.3f}")
    print(f"SHAP stability score: {np.mean(results['stability']['shap']):.3f}")
    print(f"AME stability score: {np.mean(results['stability']['ame']):.3f}")
    print(f"Average SHAP word drop impact: {np.mean(results['word_drop_impact']['shap']):.3f}")
    print(f"Average AME word drop impact: {np.mean(results['word_drop_impact']['ame']):.3f}")

if __name__ == "__main__":
    main()