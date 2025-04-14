# attribution_comparison.py
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from tqdm.auto import tqdm
import time
from llm_handler import LLM_Handler
from Attributer import Attributer
import os
from dotenv import load_dotenv


class AttributionBenchmark:
    def __init__(self,
        model_name: str = "meta-llama/Llama-3.2-1B",
        dataset_name: str = "ag_news",
        num_samples: int = 1000):
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
        print(dataset)
        # Take random sample
        indices = np.random.choice(len(dataset), self.num_samples, replace=False)
        self.data = [dataset[int(i)] for i in indices]
        print(self.data)
        if self.dataset_name =="ag_news":
            self.class_labels = ["World", "Sports", "Business", "Sci/Tech"]
        

        


if __name__ == "__main__":
    attBenchmark = AttributionBenchmark(model_name="meta-llama/Llama-3.2-1B", dataset_name= "ag_news", num_samples=10)













# class AttributionComparison:
#     def __init__(
#         self,
#         model_name: str = "meta-llama/Llama-3.2-1B",
#         dataset_name: str = "ag_news",
#         num_samples: int = 100,
#         nsamples_shap: int = 100
#     ):
#         """
#         Initialize attribution comparison experiment.
        
#         Args:
#             model_name: Name of the LLM model to use
#             dataset_name: Name of the dataset to use
#             num_samples: Number of samples to evaluate
#             nsamples_shap: Number of samples for SHAP computation
#         """
#         self.model_name = model_name
#         self.dataset_name = dataset_name
#         self.num_samples = num_samples
#         self.nsamples_shap = nsamples_shap
        
#         # Initialize LLM handler and attributer
#         self.llm_handler = LLM_Handler(
#             class_labels=["World", "Sports", "Business", "Sci/Tech"],
#             hf_auth_token=os.environ.get("HF_API_KEY"),
#             model_name=model_name
#         )
#         self.attributer = Attributer(model_name=model_name)
        
#         # Load dataset
#         self._load_dataset()
        
#     def _load_dataset(self):
#         """Load and prepare the dataset."""
#         dataset = load_dataset(self.dataset_name, split="train")
#         # Take random sample
#         indices = np.random.choice(len(dataset), self.num_samples, replace=False)
#         self.data = [dataset[i] for i in indices]
        
#     def compute_top_k_importance(self, attributions: List[Tuple[str, float]], k: int) -> List[str]:
#         """Get top k most important words based on attribution scores."""
#         sorted_attrs = sorted(attributions, key=lambda x: abs(x[1]), reverse=True)
#         return [word for word, _ in sorted_attrs[:k]]
    
#     def compute_word_drop_impact(
#         self,
#         text: str,
#         top_k_words: List[str],
#         original_label: str
#     ) -> float:
#         """Compute impact of removing top k words on classification probability."""
#         # Remove top k words
#         words = text.split()
#         filtered_words = [w for w in words if w not in top_k_words]
#         filtered_text = " ".join(filtered_words)
        
#         # Get new classification probability
#         metrics = self.llm_handler.get_classification_metrics(filtered_text, original_label)
#         return metrics['normalized_prob_given_label']
    
#     def compute_spearman_correlation(
#         self,
#         shap_attrs: List[Tuple[str, float]],
#         ame_attrs: List[Tuple[str, float]]
#     ) -> float:
#         """Compute Spearman correlation between SHAP and AME attributions."""
#         # Create word to score mapping
#         shap_scores = {word: score for word, score in shap_attrs}
#         ame_scores = {word: score for word, score in ame_attrs}
        
#         # Get common words
#         common_words = set(shap_scores.keys()) & set(ame_scores.keys())
        
#         # Create score arrays
#         shap_array = [shap_scores[word] for word in common_words]
#         ame_array = [ame_scores[word] for word in common_words]
        
#         # Compute correlation
#         correlation, _ = spearmanr(shap_array, ame_array)
#         return correlation
    
#     def run_comparison(self) -> Dict:
#         """Run the full comparison experiment."""
#         results = {
#             'top_k_importance': {'shap': [], 'ame': []},
#             'word_drop_impact': {'shap': [], 'ame': []},
#             'computation_time': {'shap': [], 'ame': []},
#             'spearman_correlation': [],
#             'stability': {'shap': [], 'ame': []}
#         }
        
#         for sample in tqdm(self.data, desc="Processing samples"):
#             text = sample['text']
#             true_label = sample['label']
            
#             # Get predicted label
#             pred_label = self.llm_handler.get_predicted_class(text)
            
#             # Compute SHAP attributions
#             start_time = time.time()
#             shap_attrs = self.attributer.attribute_shap(
#                 text, pred_label, nsamples=self.nsamples_shap
#             )
#             shap_time = time.time() - start_time
            
#             # Compute AME attributions
#             start_time = time.time()
#             ame_attrs = self.attributer.attribute(text, num_datasets=100)
#             ame_time = time.time() - start_time
            
#             # Store computation times
#             results['computation_time']['shap'].append(shap_time)
#             results['computation_time']['ame'].append(ame_time)
            
#             # Compute top-k importance
#             for k in [3, 5, 10]:
#                 shap_top_k = self.compute_top_k_importance(shap_attrs, k)
#                 ame_top_k = self.compute_top_k_importance(ame_attrs, k)
                
#                 results['top_k_importance']['shap'].append(shap_top_k)
#                 results['top_k_importance']['ame'].append(ame_top_k)
                
#                 # Compute word drop impact
#                 shap_impact = self.compute_word_drop_impact(text, shap_top_k, pred_label)
#                 ame_impact = self.compute_word_drop_impact(text, ame_top_k, pred_label)
                
#                 results['word_drop_impact']['shap'].append(shap_impact)
#                 results['word_drop_impact']['ame'].append(ame_impact)
            
#             # Compute Spearman correlation
#             correlation = self.compute_spearman_correlation(shap_attrs, ame_attrs)
#             results['spearman_correlation'].append(correlation)
            
#             # Compute stability (run each method twice)
#             shap_attrs2 = self.attributer.attribute_shap(
#                 text, pred_label, nsamples=self.nsamples_shap
#             )
#             ame_attrs2 = self.attributer.attribute(text, num_datasets=100)
            
#             stability_shap = self.compute_spearman_correlation(shap_attrs, shap_attrs2)
#             stability_ame = self.compute_spearman_correlation(ame_attrs, ame_attrs2)
            
#             results['stability']['shap'].append(stability_shap)
#             results['stability']['ame'].append(stability_ame)
        
#         return results
    
#     def visualize_results(self, results: Dict):
#         """Create visualizations of the comparison results."""
#         # 1. Computation Time Comparison
#         plt.figure(figsize=(10, 6))
#         plt.boxplot([
#             results['computation_time']['shap'],
#             results['computation_time']['ame']
#         ], labels=['SHAP', 'AME'])
#         plt.title('Computation Time Comparison')
#         plt.ylabel('Time (seconds)')
#         plt.savefig('computation_time.png')
#         plt.close()
        
#         # 2. Spearman Correlation Distribution
#         plt.figure(figsize=(10, 6))
#         plt.hist(results['spearman_correlation'], bins=20)
#         plt.title('Distribution of Spearman Correlations')
#         plt.xlabel('Correlation Coefficient')
#         plt.ylabel('Count')
#         plt.savefig('spearman_correlation.png')
#         plt.close()
        
#         # 3. Stability Comparison
#         plt.figure(figsize=(10, 6))
#         plt.boxplot([
#             results['stability']['shap'],
#             results['stability']['ame']
#         ], labels=['SHAP', 'AME'])
#         plt.title('Attribution Stability Comparison')
#         plt.ylabel('Stability Score')
#         plt.savefig('stability.png')
#         plt.close()
        
#         # 4. Word Drop Impact Comparison
#         plt.figure(figsize=(10, 6))
#         plt.boxplot([
#             results['word_drop_impact']['shap'],
#             results['word_drop_impact']['ame']
#         ], labels=['SHAP', 'AME'])
#         plt.title('Word Drop Impact Comparison')
#         plt.ylabel('Classification Probability Drop')
#         plt.savefig('word_drop_impact.png')
#         plt.close()

# def main():
#     # Initialize experiment
#     experiment = AttributionComparison(
#         model_name="meta-llama/Llama-3.2-1B",
#         num_samples=100,
#         nsamples_shap=100
#     )
    
#     # Run comparison
#     print("Running attribution comparison...")
#     results = experiment.run_comparison()
    
#     # Visualize results
#     print("Creating visualizations...")
#     experiment.visualize_results(results)
    
#     # Print summary statistics
#     print("\nSummary Statistics:")
#     print(f"Average SHAP computation time: {np.mean(results['computation_time']['shap']):.2f}s")
#     print(f"Average AME computation time: {np.mean(results['computation_time']['ame']):.2f}s")
#     print(f"Average Spearman correlation: {np.mean(results['spearman_correlation']):.3f}")
#     print(f"SHAP stability score: {np.mean(results['stability']['shap']):.3f}")
#     print(f"AME stability score: {np.mean(results['stability']['ame']):.3f}")
#     print(f"Average SHAP word drop impact: {np.mean(results['word_drop_impact']['shap']):.3f}")
#     print(f"Average AME word drop impact: {np.mean(results['word_drop_impact']['ame']):.3f}")

# if __name__ == "__main__":
#     main()