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
import re # Import re for word splitting
from scipy.stats import spearmanr

# Import necessary libraries for SHAP and IG
import shap
from captum.attr import IntegratedGradients, LayerIntegratedGradients, visualization as viz
from captum.attr._utils.visualization import format_word_importances # For better text viz
import matplotlib
# 1) If you ever use any matplotlib plots (e.g. bar charts), force Agg on headless:
matplotlib.use("Agg")  # non‑GUI backend :contentReference[oaicite:0]{index=0}


class Attributer:
    def __init__(self, llm_handler: LLM_Handler):  # Use an accessible model
        """Initialize the model and tokenizer."""
        if llm_handler != None:
            self.LLM_Handler = llm_handler
        else:
            load_dotenv('/home/hrahmani/CausalContextAttributer/config.env')
            hf_auth_token  = os.environ.get("HF_API_KEY")
            class_labels = ["Technology", "Politics", "Sports", "Art", "Other"]
            self.LLM_Handler = LLM_Handler(class_labels, hf_auth_token, "meta-llama/Llama-3.2-1B")

        
 

    def attribute_ame(self, original_prompt, num_datasets = 2000, split_by = "word", mode = "classification", save_data = False):
        """
        Compute attributions and return as (word, score) tuples.
        
        Args:
            original_prompt: Input text to analyze
            num_datasets: Number of samples for AME
            split_by: How to split text ("word" or "sentence")
            mode: Type of task ("classification")
            method_name: Attribution method ('lasso' or other)
            
        Returns:
            List[Tuple[str, float]]: List of (word, attribution_score) tuples
        """
        if mode == "classification":
            original_label =self.LLM_Handler.get_predicted_class(original_prompt)
            prompt_gen = context_processor.EfficientPromptGenerator(LLM_handler=self.LLM_Handler, prompt=original_prompt, num_datasets=num_datasets)
            X, partition = prompt_gen.create_X(split_by=split_by, mode= "1/p featurization")
            y, y_lognormalized, outputs = prompt_gen.create_y(prompt_gen.sample_prompts, original_label=original_label)
            if save_data:
                np.save(f'CausalContextAttributer/data/correct_X_{original_prompt[:5]}continue.npy', X)
                np.save(f'CausalContextAttributer/data/corrrect_y_{original_prompt[:5]}continue.npy', y_lognormalized)
            
            lasso_solver = Solver.LassoSolver(coef_scaling= prompt_gen.coef_scaling())
            coefficients = lasso_solver.fit(X, y_lognormalized)
            lasso_source_attributions = list(zip(partition.parts, coefficients))
            
            print("Lasso Coefficients as (source, score) pairs:", lasso_source_attributions)
            
            
            print('we are doing the orthogonal method')
            
            all_results = Solver.estimate_all_treatments(X, y_lognormalized)
            ortho_result = all_results['econml']
            ortho_source_attributions = []
            for i, source in enumerate(partition.parts):
                treatment_key = f'treatment_{i}'
                if treatment_key in ortho_result:
                    ate_score = ortho_result[treatment_key]['ate']
                    ortho_source_attributions.append((source, ate_score))            
             
            return (lasso_source_attributions, ortho_source_attributions)


    def attribute(self, original_prompt, num_datasets = 2000, split_by = "word", mode = "classification", method_name = 'lasso', save_data = False):
        """
        Compute attributions and return as (word, score) tuples.
        
        Args:
            original_prompt: Input text to analyze
            num_datasets: Number of samples for AME
            split_by: How to split text ("word" or "sentence")
            mode: Type of task ("classification")
            method_name: Attribution method ('lasso' or other)
            
        Returns:
            List[Tuple[str, float]]: List of (word, attribution_score) tuples
        """
        if mode == "classification":
            original_label =self.LLM_Handler.get_predicted_class(original_prompt)
            prompt_gen = context_processor.EfficientPromptGenerator(LLM_handler=self.LLM_Handler, prompt=original_prompt, num_datasets=num_datasets)
            X, partition = prompt_gen.create_X(split_by="word", mode= "1/p featurization")
            y, y_lognormalized, outputs = prompt_gen.create_y(prompt_gen.sample_prompts, original_label=original_label)
            if save_data:
                np.save(f'CausalContextAttributer/data/correct_X_{original_prompt[:5]}continue.npy', X)
                np.save(f'CausalContextAttributer/data/corrrect_y_{original_prompt[:5]}continue.npy', y_lognormalized)
            if method_name =='lasso':
                lasso_solver = Solver.LassoSolver(coef_scaling= prompt_gen.coef_scaling())
                coefficients = lasso_solver.fit(X, y_lognormalized)
                lasso_word_attributions = list(zip(partition.parts, coefficients))
                
                print("Lasso Coefficients as (word, score) pairs:", lasso_word_attributions)
                return lasso_word_attributions
            
            elif method_name =='orthogonal':
                print('we are doing the orthogonal method')
                all_results = Solver.estimate_all_treatments(X, y_lognormalized)
                ortho_word_attributions = []
                for i, word in enumerate(partition.parts):
                    treatment_key = f'treatment_{i}'
                    if treatment_key in all_results:
                        ate_score = all_results[treatment_key]['ate']
                        ortho_word_attributions.append((word, ate_score))
                
                return ortho_word_attributions
            
            else:
                lasso_solver = Solver.LassoSolver(coef_scaling= prompt_gen.coef_scaling())
                coefficients = lasso_solver.fit(X, y_lognormalized)
                lasso_word_attributions = list(zip(partition.parts, coefficients))

                all_results = Solver.estimate_all_treatments(X, y_lognormalized)
                ortho_word_attributions = []
                for i, word in enumerate(partition.parts):
                    treatment_key = f'treatment_{i}'
                    if treatment_key in all_results:
                        ate_score = all_results[treatment_key]['ate']
                        ortho_word_attributions.append((word, ate_score))
                
                return (lasso_word_attributions, ortho_word_attributions)


                 


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

    def attribute_shap(self, text: str, target_label: str, nsamples: int = 100) -> List[Tuple[str, float]]:
        """
        Computes word-level attributions using SHAP's Explainer with custom tokenizer.
        
        Args:
            text: Input text to analyze
            target_label: Target class label to explain
            
        Returns:
            List of (word, attribution) tuples
        """
        print(f"\n--- Running SHAP Attribution for label: {target_label} ---")

        # Define prediction function
        def predict_fn(texts, mode = 'prob'):
            """Wrapper for model predictions."""
            scores = []
            for text in texts:
                metrics = self.LLM_Handler.get_classification_metrics(text, target_label)
                # Use log probabilities for more stable SHAP values
                if mode == 'prob':
                    prob = metrics.get('normalized_prob_given_label')
                    scores.append(float(prob))
                elif mode =='log':
                    log_prob = metrics.get('normalized_log_prob_given_label')
                    scores.append(float(log_prob))
            return np.array(scores).reshape(-1, 1)
        
        try:
            masker = shap.maskers.Text(r"\W")  # masker that split on non words. So we have list of words.
            explainer = shap.Explainer(
                predict_fn,
                masker,
                output_names=[target_label],
                max_evals=nsamples
            )
            
            # Compute SHAP values
            shap_values = explainer([text])

            # print(f'shapley values{shap_values}')
            words = shap_values.data[0]
            attributions = shap_values.values[0]
            
            # Match words with their attributions
            word_attributions = []
            for word, attr in zip(words, attributions):
                clean = self.clean_word(word)
                if clean:  # Only include non-empty
                    word_attributions.append((clean, float(attr)))

            return word_attributions
            
        except Exception as e:
            print(f"Error in SHAP attribution: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def compute_spearman_correlation(
        self,
        shap_attrs: List[Tuple[str, float]],
        ame_attrs: List[Tuple[str, float]]
    ) -> float:
        """Compute Spearman correlation between SHAP and AME attributions."""
        # Create word to score mapping
        shap_scores = {word: score for word, score in shap_attrs}
        ame_scores = {word: score for word, score in ame_attrs}
        
        # Get common words
        common_words = set(shap_scores.keys()) & set(ame_scores.keys())
        # print(f'common words are: {common_words}')
        
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
        original_label = self.LLM_Handler.get_predicted_class(text)
        original_metrics = self.LLM_Handler.get_classification_metrics(text, original_label)
        original_log_prob = original_metrics['normalized_log_prob_given_label']
        print(original_log_prob)
        # Sort words by absolute attribution score
        sorted_words = sorted(attributions, key=lambda x: abs(x[1]), reverse=True)
        words_to_remove = [word for word, _ in sorted_words[:k]]
        
        # Create modified text by replacing top k words with __
        modified_text = text
        for word in words_to_remove:
            modified_text = re.sub(r'\b' + re.escape(word) + r'\b', '__', modified_text)
        
        print(modified_text)
        # Get new prediction and probability
        modified_metrics = self.LLM_Handler.get_classification_metrics(modified_text, original_label)
        modified_log_prob = modified_metrics['normalized_log_prob_given_label']
        print(modified_log_prob)
        
        return original_log_prob - modified_log_prob

    def clean_word(self, word):
        # Remove leading/trailing whitespace and punctuation
        return re.sub(r"^\W+|\W+$", "", word).strip()



if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.2-1B"
    load_dotenv('/home/hrahmani/CausalContextAttributer/config.env')
    hf_auth_token  = os.environ.get("HF_API_KEY")
    class_labels = ["Technology", "Politics", "Sports", "Art", "Other"]
    llm_handler = LLM_Handler(class_labels, hf_auth_token, model_name)
    attributer = Attributer(llm_handler=llm_handler)
    text = "Local Mayor Launches Initiative to enhance urban public transport."
    target_label = "Politics"
        # Get attributions as (word, score) tuples
    # ame_lasso_attributions = attributer.attribute(text, num_datasets=1000, method_name='lasso')

    # # Print results
    # print("\nAME lasso Attributions:")
    # for word, score in ame_lasso_attributions:
    #     print(f"{word}: {score:.4f}")
    
    # print("by AME lasso normalized log probability drops by", attributer.compute_log_prob_drop(text, ame_lasso_attributions, k=1))
    ame_lasso_attributions, ame_ortho_attributions = attributer.attribute_ame(text, num_datasets=1000)

    # Print results
    print("\nAME lasso Attributions:")
    for word, score in ame_lasso_attributions:
        print(f"{word}: {score:.4f}")
    
    print("by AME lasso normalized log probability drops by", attributer.compute_log_prob_drop(text, ame_lasso_attributions, k=1))

    print("\nAME orthogonal Attributions:")
    for word, score in ame_ortho_attributions:
        print(f"{word}: {score:.4f}")
    
    print("by AME Orthogonal normalized log probability drops by", attributer.compute_log_prob_drop(text, ame_ortho_attributions, k=1))

    print(f'the spaersman correlation between lasso and orthogonal {attributer.compute_spearman_correlation(shap_attrs=ame_ortho_attributions, ame_attrs=ame_lasso_attributions)}')

    shap_attributions = attributer.attribute_shap(text, target_label, nsamples=1000)
    print("\nSHAP Attributions:")
    for word, score in shap_attributions:
        print(f"{word}: {score:.4f}")

    print(f'the spaersman correlation between shap and ame_lasso{attributer.compute_spearman_correlation(shap_attrs=shap_attributions, ame_attrs=ame_lasso_attributions)}')

    
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