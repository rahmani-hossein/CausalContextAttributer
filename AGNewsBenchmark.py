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
        original_label = self.llm_handler.get_predicted_class(text)
        original_metrics = self.llm_handler.get_classification_metrics(text, original_label)
        original_log_prob = original_metrics['normalized_log_prob_given_label']
        
        sorted_words = sorted(attributions, key=lambda x: abs(x[1]), reverse=True)
        words_to_remove = [word for word, _ in sorted_words[:k]]
        
        # Create modified text by replacing top k words with __
        modified_text = text
        for word in words_to_remove:
            modified_text = re.sub(r'\b' + re.escape(word) + r'\b', '__', modified_text)
        
        modified_label = self.llm_handler.get_predicted_class(modified_text)
        modified_metrics = self.llm_handler.get_classification_metrics(modified_text, original_label)
        modified_log_prob = modified_metrics['normalized_log_prob_given_label']
        
        # return original_log_prob - modified_log_prob
        return {
            'original_label': original_label,
            'original_log_prob': original_log_prob,
            'modified_log_prob': modified_log_prob,
            'log_prob_drop': original_log_prob - modified_log_prob,
            'modified_label': modified_label,
            'removed_words': words_to_remove
        }

    def run_comparison(self, methods: List[str] = ['shap', 'lasso', 'orthogonal'], k_values: List[int] = [1, 2, 3], **kwargs) -> Dict:
        results = {}
        for method in methods:
            results[method] = {k: {
                'log_prob_drops': [],
                'label_changes': []
            } for k in k_values}
            
        for sample in tqdm(self.data,  desc=f"Evaluating each sample"):
                original_text = sample['text']
                # true_label = self.index_to_label[sample['label']]
                original_label = self.llm_handler.get_predicted_class(original_text)
                shap_attributions = self.attributer.attribute_shap(sample, original_label, nsamples=self.num_attribution_samples)
                
                lasso_attributions = self.attributer.attribute(original_text, num_datasets=self.num_attribution_samples, method_name='lasso')

                ortho_attributions = self.attributer.attribute(original_text, num_datasets=self.num_attribution_samples, method_name='orthogonal')
                # Compute metrics for each k
                for method in methods:
                    if method == 'lasso':
                        attributions = lasso_attributions
                    elif method =='shap':
                        attributions  = shap_attributions
                    else:
                        attributions  = ortho_attributions
                    for k in k_values:
                        metrics = self.compute_log_prob_drop(original_text, attributions, k)
                        
                        results[method][k]['log_prob_drops'].append(metrics['log_prob_drop'])
                        results[method][k]['label_changes'].append(
                            metrics['original_label'] != metrics['modified_label'])
        # Compute summary statistics
        summary = {}
        for method in methods:
            summary[method] = {}
            for k in k_values:
                drops = results[method][k]['log_prob_drops']
                changes = results[method][k]['label_changes']
                
                summary[method][k] = {
                    'mean_log_prob_drop': np.mean(drops),
                    'std_log_prob_drop': np.std(drops),
                    'label_change_rate': np.mean(changes),
                }
        
        return summary
    

    def plot_results(self, summary: Dict):
        """Create plots similar to the paper figure."""
        # Create results directory if it doesn't exist
        import os
        from datetime import datetime
        
        # Create timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = "attribution_results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Plot log probability drops
        plt.figure(figsize=(12, 6))
        
        # Prepare data for plotting
        methods = list(summary.keys())
        k_values = list(summary[methods[0]].keys())
        x_positions = np.arange(len(k_values))
        width = 0.25
        
        # Use a color map for better visualization
        colors = ['#2ecc71', '#3498db', '#e74c3c']  # Green, Blue, Red
        
        # Plot bars for each method
        for i, (method, color) in enumerate(zip(methods, colors)):
            means = [summary[method][k]['mean_log_prob_drop'] for k in k_values]
            stds = [summary[method][k]['std_log_prob_drop'] for k in k_values]
            
            plt.bar(x_positions + i*width, means, width, 
                label=method.upper(),
                color=color,
                yerr=stds, capsize=5)
        
        plt.xlabel('Top-k Words Removed', fontsize=12)
        plt.ylabel('Log Probability Drop', fontsize=12)
        plt.title('Impact of Removing Top-k Words by Attribution Method', fontsize=14, pad=20)
        plt.xticks(x_positions + width, [f'k={k}' for k in k_values], fontsize=10)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save with timestamp
        filename = os.path.join(results_dir, f'log_prob_drops_{timestamp}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot label change rates
        plt.figure(figsize=(12, 6))
        
        for i, (method, color) in enumerate(zip(methods, colors)):
            label_changes = [summary[method][k]['label_change_rate'] for k in k_values]
            
            plt.bar(x_positions + i*width, label_changes, width,
                label=method.upper(),
                color=color)
        
        plt.xlabel('Top-k Words Removed', fontsize=12)
        plt.ylabel('Label Change Rate', fontsize=12)
        plt.title('Label Change Rate by Attribution Method', fontsize=14, pad=20)
        plt.xticks(x_positions + width, [f'k={k}' for k in k_values], fontsize=10)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save with timestamp
        filename = os.path.join(results_dir, f'label_changes_{timestamp}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save summary statistics to a text file
        stats_filename = os.path.join(results_dir, f'summary_stats_{timestamp}.txt')
        with open(stats_filename, 'w') as f:
            f.write(f"Attribution Analysis Results - {timestamp}\n")
            f.write("=" * 50 + "\n\n")
            
            for method in methods:
                f.write(f"\n{method.upper()} Results:\n")
                f.write("-" * 20 + "\n")
                for k in k_values:
                    f.write(f"\nTop-{k}:\n")
                    f.write(f"Log Prob Drop: {summary[method][k]['mean_log_prob_drop']:.3f} ± {summary[method][k]['std_log_prob_drop']:.3f}\n")
                    f.write(f"Label Change Rate: {summary[method][k]['label_change_rate']:.3f}\n")
                f.write("\n")

def main():
    # Initialize benchmark
    benchmark = AttributionBenchmark(
        model_name="meta-llama/Llama-3.2-1B",
        dataset_name="ag_news",
        num_samples=10,
        num_attribution_samples=1000
    )
    
    print(benchmark.data)
    # Run comparison with all three methods
    print("Running comparison...")
    summary = benchmark.run_comparison(
        methods=['shap', 'lasso', 'orthogonal'],
        k_values=[1, 2, 3]
    )
    
    # Plot and save results
    print("Creating and saving plots and statistics...")
    benchmark.plot_results(summary)
    
    print("\nResults have been saved to the 'attribution_results' directory.")

if __name__ == "__main__":
    main()
    
# if __name__ == "__main__":
#     attBenchmark = AttributionBenchmark(model_name="meta-llama/Llama-3.2-1B", dataset_name= "ag_news", num_samples=10)