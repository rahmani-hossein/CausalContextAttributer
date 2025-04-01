# Attribution and Metrics Computation
import numpy as np
import torch
from typing import List, Dict, Tuple
from dotenv import load_dotenv
import os
from llm_handler import LLM_Handler
import context_processor
import Solver
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Attributer:
    def __init__(self, model_name: str ):  # Use an accessible model
        """Initialize the model and tokenizer."""

        load_dotenv('/home/hrahmani/CausalContextAttributer/config.env')
        hf_auth_token  = os.environ.get("HF_API_KEY")
        class_labels = ["Technology", "Politics", "Sports", "Art", "Other"]
        self.LLM_Handler = LLM_Handler(class_labels, hf_auth_token, model_name)
        

    def attribute(self, original_prompt, num_datasets = 2000, split_by = "word", mode = "classification"):
        if mode == "classification":
            original_label =self.LLM_Handler.get_predicted_class(original_prompt)
            prompt_gen = context_processor.EfficientPromptGenerator(LLM_handler=self.LLM_Handler, prompt=original_prompt, num_datasets=num_datasets)
            X, partition = prompt_gen.create_X(split_by="word", mode= "1/p featurization")
            y, y_lognormalized, outputs = prompt_gen.create_y(prompt_gen.sample_prompts, original_label=original_label)
            np.save('CausalContextAttributer/data/correct_X_pfeat.npy', X)
            np.save('CausalContextAttributer/data/corrrect_y.npy', y)
            lasso_solver = Solver.LassoSolver(coef_scaling= prompt_gen.coef_scaling())
            res = lasso_solver.fit(X, y)
            print("Lasso Coefficients:", res)
              # Get results for all treatments
            all_results = Solver.estimate_all_treatments(X, y)
            
            # Print summary
            print("\nResults Summary:")
            print("-" * 50)
            for treatment, results in all_results.items():
                print(f"\n{treatment}:")
                print(f"ATE: {results['ate']:.3f}")
                print(f"Final Loss: {results['final_loss']:.4f}")
                print(f"CATE std: {np.std(results['cate_estimates']):.3f}")

        return results['ate']
    


    def plot_logprob_distributions(self, X, y, word_labels=None, save_path='logprob_distributions.png'):
        """
        Create a box plot for each word, showing the distribution of log probabilities
        for sample prompts with and without the word.

        Parameters:
        - X (np.ndarray): Binary matrix [n_samples, n_words], 1 if word is present, 0 if absent.
        - y (np.ndarray): Vector [n_samples], log probabilities of each sample prompt.
        - word_labels (list, optional): List of word names. If None, uses "Word 0", "Word 1", etc.
        """

        # Set default word labels if none provided
        if word_labels is None:
            word_labels = [f"Word {i}" for i in range(X.shape[1])]
        assert len(word_labels) == X.shape[1], "Number of word labels must match number of columns in X"

        data = []
        for j, word in enumerate(word_labels):
            # Indices where the word is present
            idx_with = np.where(X[:, j] > 0)[0]
            # Indices where the word is absent
            idx_without = np.where(X[:, j] <= 0)[0]

            # Add log probabilities for 'with' group
            if len(idx_with) > 0:
                for val in y[idx_with]:
                    data.append({'word': word, 'group': 'with', 'log_prob': val})
            else:
                print(f"Warning: No samples with '{word}' present.")

            # Add log probabilities for 'without' group
            if len(idx_without) > 0:
                for val in y[idx_without]:
                    data.append({'word': word, 'group': 'without', 'log_prob': val})
            else:
                print(f"Warning: No samples without '{word}' present.")

        
        # Convert to DataFrame for Seaborn
        df = pd.DataFrame(data)

        # Create the box plot
        plt.figure(figsize=(12, 6))  # Adjust size as needed
        sns.boxplot(data=df, x='word', y='log_prob', hue='group', showmeans=True,
                    meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"})
        plt.xticks(rotation=45)  # Rotate x-axis labels for readability
        plt.title("Log Probability Distributions With and Without Each Word")
        plt.xlabel("Word")
        plt.ylabel("Log Probability")
        plt.legend(title="Word Presence")
        plt.tight_layout()
        plt.savefig(save_path)  # Save the plot to the specified file
        print(f"Plot saved to {save_path}")
        plt.close()  # Close the figure to free memory



if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.2-1B"
    
    attributer = Attributer(model_name=model_name)
    original_prompt = "Local Mayor Launches Initiative to Enhance Urban Public Transport."
    original_label =attributer.LLM_Handler.get_predicted_class(original_prompt)
    prompt_gen = context_processor.EfficientPromptGenerator(LLM_handler=attributer.LLM_Handler, prompt=original_prompt, num_datasets=100)
    X, partition = prompt_gen.create_X(split_by="word", mode= "1/p featurization")
    y, y_lognormalized, outputs = prompt_gen.create_y(prompt_gen.sample_prompts, original_label)
    m = X.shape[0]
    for i in range(m):
        print(f'for sampled prompt {i} with prob {prompt_gen.ps[i]}: {prompt_gen.sample_prompts[i]} got classification answer {outputs[i]} with logprob {y[i]} with normalized logprob {y_lognormalized[i]}')

    attributer.plot_logprob_distributions(X,y, word_labels= partition.parts, save_path='CausalContextAttributer/data/logprob_distributions.png')

    # attributer.attribute(original_prompt=original_prompt, num_datasets=2000)











# # Example Usage
# if __name__ == "__main__":
#     # Note: Llama 3-8B may require special access; using Llama 2-7B as a placeholder     meta-llama/Meta-Llama-3-8B
#     calc = AttributionCalculator(model_name="meta-llama/Llama-3.2-1B")  # Adjust as needed

#     # Single data point
#     headline = "Election Results Announced Today"
#     true_label = "Politics"
#     import time
#     start_time = time.time()
#     label = calc.LLM_Handler.get_predicted_class(headline)
#     end_time = time.time()
#     time_taken = end_time - start_time
#     print(f"Time taken for llm call: {time_taken} seconds")
#     print(label)

#     print(calc.LLM_Handler.getClassification_log_prob("__ Results Announced Today", label))


#     headline2 = " __ Results Announced Today"
#     true_label = "Politics"
#     label2 = calc.LLM_Handler.get_predicted_class(headline2)
#     print(label2)
#     print(calc.LLM_Handler.getClassification_log_prob(headline2, label2))
#     # Dataset
#     dataset = [
#         ("Election Results Announced Today", "politics"),
#         ("Football Match Ends in Draw", "sports"),
#         ("New Movie Released This Week", "entertainment")
#     ]
    # metrics = calc.evaluate_dataset(dataset, max_k=3)
    # print("\nDataset Metrics:")
    # for metric, value in metrics.items():
    #     print(f"{metric}: {value}")
