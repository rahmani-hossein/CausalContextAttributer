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
import os
import re # Import re for word splitting
import math

# Import necessary libraries for SHAP and IG
import shap
from captum.attr import IntegratedGradients, LayerIntegratedGradients, visualization as viz
from captum.attr._utils.visualization import format_word_importances # For better text viz
from datasets import load_dataset # To load a standard dataset

import matplotlib
# 1) If you ever use any matplotlib plots (e.g. bar charts), force Agg on headless:
matplotlib.use("Agg")  # non‑GUI backend :contentReference[oaicite:0]{index=0}


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

    
    # --- Integrated Gradients Attribution Method ---
    def attribute_integrated_gradients(self, text: str, target_label: str) -> List[Tuple[str, float]]:
        """Computes word-level attributions using Integrated Gradients."""
        print(f"\n--- Running Integrated Gradients Attribution for label: {target_label} ---")

        # 1. Get the target token ID for the predicted class
        target_token_id = self.LLM_Handler.class_tokens.get(target_label)
        if target_token_id is None:
            print(f"Error: Could not find token ID for label '{target_label}'")
            return []
        print(f"Target Label: {target_label}, Target Token ID: {target_token_id}")

        # 2. Define the forward function for Captum
        #    Input: token embeddings (or input_ids)
        #    Output: Logit of the target class token *before* softmax
        def model_forward(inputs_embeds):
            # Construct the prompt text around the input
            # This is tricky with embeddings. A common approach is to get logits directly from input_ids.
            # Let's redefine to work with input_ids first, then adapt if needed for LayerIG on embeddings.

            # Alternate forward func using input_ids (simpler for basic IG)
            # Assumes 'inputs' are input_ids
            input_ids = inputs_embeds # Rename for clarity if passing IDs

            # We need the full prompt structure for the model
            # This requires modifying the input_ids to include the prompt template
            # This is complex with IG baselines. Let's stick to LayerIG on embeddings.

            # Revert to Embeddings - Define forward for LayerIntegratedGradients
            # inputs_embeds: embeddings of the input text tokens
            # We need the full sequence including prompt template tokens for the model context

            # Get embeddings for the prompt template
            prompt_prefix = f"Classify this headline: "
            prefix_tokens = self.tokenizer(prompt_prefix, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
            prefix_embeds = self.model.get_input_embeddings()(prefix_tokens) # Shape: (1, seq_len_prefix, embed_dim)

            prompt_suffix = ". The category is: "
            suffix_tokens = self.tokenizer(prompt_suffix, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
            suffix_embeds = self.model.get_input_embeddings()(suffix_tokens) # Shape: (1, seq_len_suffix, embed_dim)

            # Combine embeddings: prefix + input_text + suffix
            # inputs_embeds has shape (batch_size=1, seq_len_input, embed_dim)
            full_embeds = torch.cat([prefix_embeds, inputs_embeds, suffix_embeds], dim=1)

            # Pass combined embeddings through the model
            outputs = self.model(inputs_embeds=full_embeds)
            logits = outputs.logits # Shape: (batch, seq_len_full, vocab_size)
            # We need the logits for the *last* token position, which predicts the category
            last_token_logits = logits[:, -1, :] # Shape: (batch, vocab_size)
            return last_token_logits


        # 3. Prepare inputs and baseline for IG
        # Tokenize the input text *only*
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device) # Needed if model uses it

        # Get input embeddings
        input_embeddings = self.model.get_input_embeddings()(input_ids) # Shape: (1, seq_len, embed_dim)

        # Baseline: embedding of padding tokens or zeros
        baseline_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        baseline_ids = torch.full_like(input_ids, baseline_token_id)
        baseline_embeddings = self.model.get_input_embeddings()(baseline_ids)

        # 4. Initialize Integrated Gradients
        # Use LayerIntegratedGradients with the embedding layer
        lig = LayerIntegratedGradients(model_forward, self.model.get_input_embeddings())

        # 5. Compute attributions
        try:
            attributions_ig = lig.attribute(inputs_embeds=input_embeddings,
                                            baselines=baseline_embeddings,
                                            target=target_token_id, # Target is the class token ID
                                            n_steps=50, # Number of steps for integration
                                            internal_batch_size=1) # Adjust if needed for memory
        except Exception as e:
            print(f"Error during Integrated Gradients calculation: {e}")
            return []

        # Attributions have shape (1, seq_len, embed_dim). Sum across embedding dimension.
        attributions_sum = attributions_ig.sum(dim=-1).squeeze(0)
        attributions_norm = attributions_sum / torch.norm(attributions_sum) # Normalize
        attributions_np = attributions_norm.cpu().detach().numpy()

        # 6. Map token attributions back to words
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        # Simple word splitting for mapping (same as SHAP)
        words = re.findall(r'\b\w+\b', text.lower())
        # Use token offsets for more robust mapping if available and needed
        # For now, use a basic aggregation or just show token attributions
        token_word_mapping = list(zip(tokens, attributions_np))

        # Try to aggregate to word level (basic sum)
        # Need offsets for proper aggregation. This is a placeholder.
        # word_attributions = self._aggregate_token_to_word_attributions(tokens, attributions_np, text) # Need a helper
        # Using token attributions directly for now
        print(f"Integrated Gradients Attributions (Token Level): {token_word_mapping}")

        # Optional: Visualize using Captum's tools
        # self.visualize_attributions(attributions_norm, tokens, target_label, text)

        print("--- Finished Integrated Gradients ---")
        # Return token-level attributions for now, word-level needs careful mapping
        return token_word_mapping # Or word_attributions if aggregation is implemented

    from captum.attr import visualization as viz

    def visualize_attributions(self, attributions, tokens, label, text):
        """Helper to visualize Captum attributions."""
        vis_data_records = [viz.VisualizationDataRecord(
                                attributions,
                                0, # pred_prob (dummy)
                                label, # predicted_class
                                label, # true_class (use predicted for viz)
                                "attributions", # attribution type
                                attributions.sum(), # attribution score
                                tokens, # raw_input_ids
                                1)] # convergence score (dummy)
        html_output = viz.visualize_text(vis_data_records)
        # You can save html_output.data to an HTML file or display in Jupyter
        print("Captum Visualization HTML generated (display in browser/notebook).")


    def attribute_shap(self, text: str, target_label: str) -> List[Tuple[str, float]]:
        """Computes word-level attributions using SHAP's KernelExplainer."""
        print(f"\n--- Running SHAP Attribution for label: {target_label} ---")
        words = self.LLM_Handler.tokenizer.tokenize(text)  # For display purposes
        print("Debug: Words (LLaMA tokens):", words)


        def predict_proba(text_list: np.ndarray, label, mode="prob") -> np.ndarray:
            scores = []
            print("Debug: predict_proba inputs:", text_list)
            for prompt_text in text_list:
                metrics = self.LLM_Handler.get_classification_metrics(prompt_text, label)

                if mode == "prob":
                    # Use normalized probability within the defined classes
                    probability = metrics.get('normalized_prob_given_label')
                    probability = float(probability) if probability is not None else 0.5
                    scores.append(probability)

                elif mode == "log":
                    log_probability = metrics.get('normalized_log_prob_given_label')
                    log_probability = float(log_probability) if log_probability is not None else math.log(0.5)
                    scores.append(log_probability)
            result = np.array(scores, dtype=np.float64).reshape(-1, 1)
            print("Debug: predict_proba output:", result)
            return result # Ensure 2D output (n_samples, 1)

        # Custom masking function
        def mask_text(texts, mask):
            masked_texts = []
            for t in texts:
                tokens = self.LLM_Handler.tokenizer.tokenize(t)
                # Ensure mask length matches tokens
                if len(mask) != len(tokens):
                    mask = mask[:len(tokens)]  # Truncate if needed
                masked = [tok if m else "[MASK]" for tok, m in zip(tokens, mask)]
                masked_text = self.LLM_Handler.tokenizer.convert_tokens_to_string(masked)
                masked_texts.append(masked_text)
            print("Debug: masked_texts:", masked_texts)
            return np.array(masked_texts)

        # Wrapper that applies masking
        def wrapped_predict(X):
            # X is a binary mask matrix from KernelExplainer (shape: n_samples, n_features)
            print("Debug: wrapped_predict X (mask):", X)
            masked_inputs = []
            for mask_row in X:
                # Apply mask to the original text
                masked_text = mask_text([text], mask_row)[0]
                masked_inputs.append(masked_text)
            masked_inputs = np.array(masked_inputs)
            print("Debug: wrapped_predict masked_inputs:", masked_inputs)
            return predict_proba(masked_inputs, target_label)

        try:
            # Background: original text (single sample)
            background = np.array([text])

            explainer = shap.KernelExplainer(wrapped_predict, background)
            print("Debug: Explainer created")

            # Compute SHAP values
            shap_values = explainer.shap_values(np.ones((1, len(words))), nsamples=100)
            print("Debug: shap_values:", shap_values)

            word_attributions = list(zip(words, shap_values[0]))
            print(f"SHAP Attributions: {word_attributions}")
            return word_attributions

        except Exception as e:
            print(f"Error in SHAP attribution: {str(e)}")
            return []



if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.2-1B"
    attributer = Attributer(model_name=model_name)
    text = "Local Mayor Launches Initiative to enhance urban public transport."
    target_label = "Politics"
    shap_attributions = attributer.attribute_shap(text, target_label)
    print(shap_attributions)


# if __name__ == "__main__":
#     model_name = "meta-llama/Llama-3.2-1B"
    
#     attributer = Attributer(model_name=model_name)
#     original_prompt = "Local Mayor Launches Initiative to Enhance Urban Public Transport."
#     original_label =attributer.LLM_Handler.get_predicted_class(original_prompt)
#     shap_attributions = attributer.attribute_shap(original_prompt, original_label)
    # prompt_gen = context_processor.EfficientPromptGenerator(LLM_handler=attributer.LLM_Handler, prompt=original_prompt, num_datasets=100)
    # X, partition = prompt_gen.create_X(split_by="word", mode= "1/p featurization")
    # y, y_lognormalized, outputs = prompt_gen.create_y(prompt_gen.sample_prompts, original_label)
    # m = X.shape[0]
    # for i in range(m):
    #     print(f'for sampled prompt {i} with prob {prompt_gen.ps[i]}: {prompt_gen.sample_prompts[i]} got classification answer {outputs[i]} with logprob {y[i]} with normalized logprob {y_lognormalized[i]}')

    # attributer.plot_logprob_distributions(X,y, word_labels= partition.parts, save_path='CausalContextAttributer/data/logprob_distributions.png')

    # attributer.attribute(original_prompt=original_prompt, num_datasets=2000)







# Helper function to summarize attributions across subword tokens
def summarize_attributions(attributions, token_offsets, words):
    """Aggregates token attributions to word level."""
    word_attributions = np.zeros(len(words))
    word_indices = [] # Keep track of which word each token belongs to

    current_word_idx = 0
    for i, (start, end) in enumerate(token_offsets):
        # Find the word corresponding to the token's start position
        # This is a simple approach; more robust mapping might be needed for complex tokenization
        while current_word_idx < len(words) and start >= len(" ".join(words[:current_word_idx+1])):
             current_word_idx += 1
        if current_word_idx < len(words):
             word_indices.append(current_word_idx)
        else:
             word_indices.append(-1) # Should not happen with correct offsets

    # Sum attributions for tokens belonging to the same word
    for i, word_idx in enumerate(word_indices):
        if word_idx != -1:
             # Ensure attributions tensor is on CPU and converted to numpy
             attr_val = attributions[i].cpu().numpy() if isinstance(attributions, torch.Tensor) else attributions[i]
             word_attributions[word_idx] += attr_val

    return list(zip(words, word_attributions))


# # --- Main Execution Block ---
# if __name__ == "__main__":
#     model_name = "meta-llama/Llama-3.2-1B" # Or your chosen Llama model
#     attributer = Attributer(model_name=model_name)

#     # --- Load Dataset ---
#     print("\nLoading dataset...")
#     # Use AG News dataset, take a small sample from the test set
#     try:
#         # Load only a few samples to test
#         num_samples_to_test = 3
#         ag_news_dataset = load_dataset("ag_news", split=f'test[:{num_samples_to_test}]')
#         samples = [{"text": item["text"]} for item in ag_news_dataset] # Extract text
#         print(f"Loaded {len(samples)} samples from AG News.")
#     except Exception as e:
#         print(f"Failed to load dataset: {e}")
#         # Fallback sample if dataset loading fails
#         samples = [
#             {"text": "Local Mayor Launches Initiative to Enhance Urban Public Transport."},
#             {"text": "FC Barcelona secures victory in the final match of the season."},
#             {"text": "New AI discovers patterns in ancient star formations."},
#             {"text": "Renowned artist unveils controversial new sculpture downtown."},
#         ]
#         print("Using fallback samples.")


#     # --- Process Each Sample ---
#     for i, sample in enumerate(samples):
#         original_prompt = sample["text"]
#         print(f"\n===== Processing Sample {i+1} =====")
#         print(f"Original Text: {original_prompt}")

#         # 1. Get the model's predicted label from the defined categories
#         predicted_label = attributer.LLM_Handler.get_predicted_class(original_prompt)
#         print(f"Predicted Label: {predicted_label}")

#         # Check if prediction is valid before proceeding
#         if predicted_label not in attributer.class_labels:
#             print(f"Warning: Predicted label '{predicted_label}' not in defined class labels. Skipping attributions.")
#             continue

#         # 2. Run your original attribution method (if desired)
#         #    Note: This returns ATE/Lasso coeffs, not directly comparable word scores like SHAP/IG
#         #    Adjust num_datasets for speed during testing
#         try:
#             # Reducing num_datasets significantly for faster testing
#             user_method_results = attributer.attribute(original_prompt, num_datasets=100)
#             # print(f"User Method Results (Lasso Coefs): {user_method_results}")
#         except Exception as e:
#             print(f"Error running original attribution method: {e}")

#         # 3. Run SHAP attribution
#         try:
#             shap_results = attributer.attribute_shap(original_prompt, predicted_label)
#             # print(f"SHAP Results: {shap_results}")
#         except Exception as e:
#             print(f"Error running SHAP attribution: {e}")


#         # 4. Run Integrated Gradients attribution
#         try:
#             ig_results = attributer.attribute_integrated_gradients(original_prompt, predicted_label)
#             # print(f"Integrated Gradients Results (Token Level): {ig_results}")
#         except Exception as e:
#             print(f"Error running Integrated Gradients attribution: {e}")

#         print(f"===== Finished Sample {i+1} =====")

#     print("\nComparison Finished.")