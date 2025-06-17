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
import shap
import matplotlib
# 1) If you ever use any matplotlib plots (e.g. bar charts), force Agg on headless:
matplotlib.use("Agg")  # non‑GUI backend :contentReference[oaicite:0]{index=0}
import nltk
from spacy.lang.en import English
from typing import Callable, List, Tuple, Dict, Union
from datetime import datetime


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
        Compute attributions and return as a dictionary containing partition, lasso coefficients, and ortho coefficients.
        
        Args:
            original_prompt: Input text to analyze
            num_datasets: Number of samples for AME
            split_by: How to split text ("word" or "sentence")
            mode: Type of task ("classification")
            method_name: Attribution method ('lasso' or other)
            
        Returns:
            Dict with keys:
                - partition: list of parts (words or sentences)
                - lasso_coefficients: list of floats
                - ortho_coefficients: list of floats
        """
        if mode == "classification":
            original_label = self.LLM_Handler.get_predicted_class(original_prompt)
            prompt_gen = context_processor.EfficientPromptGenerator(LLM_handler=self.LLM_Handler, prompt=original_prompt, num_datasets=num_datasets)
            X, partition = prompt_gen.create_X(split_by=split_by, mode= "1/p featurization")
            y, y_lognormalized, outputs = prompt_gen.create_y(prompt_gen.sample_prompts, original_label=original_label)
            if save_data:
                np.save(f'CausalContextAttributer/data/correct_X_{original_prompt[:5]}continue.npy', X)
                np.save(f'CausalContextAttributer/data/corrrect_y_{original_prompt[:5]}continue.npy', y_lognormalized)
            
            lasso_solver = Solver.LassoSolver(coef_scaling= prompt_gen.coef_scaling())
            lasso_coefficients = lasso_solver.fit(X, y_lognormalized)
            # lasso_source_attributions = list(zip(partition.parts, lasso_coefficients))
            all_results = Solver.estimate_all_treatments(X, y_lognormalized)
            # i= 2
            # j=3
            # print(f' the joint ATEof the source {i} and {j} is: (three different methods ){Solver.calculate_joint_ate(all_results=all_results, i = i, j = j, n_sources=X.shape[1])}')
            ortho_result = all_results['econml']
            ortho_coefficients = []
            ortho_cate_attributions = []
            for i, source in enumerate(partition.parts):
                treatment_key = f'treatment_{i}'
                if treatment_key in ortho_result:
                    ate_score = ortho_result[treatment_key]['ate']
                    ortho_coefficients.append(ate_score)
                    ortho_cate_attributions.append(ortho_result[treatment_key]['cate_attributions'])
                else:
                    print(f'{source} not in ortho_result')
                    ortho_coefficients.append(np.nan)
            return {
                "partition": partition.parts,
                "lasso_coefficients": lasso_coefficients,
                "ortho_coefficients": ortho_coefficients,
                "ortho_cate_attributions": ortho_cate_attributions
            }



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

    def attribute_shap(self, text: str, target_label: str, nsamples: int = 100, split_by: str = "word") -> List[Tuple[str, float]]:
        """
        Computes source (word or sentence) attributions using SHAP's Explainer with custom tokenizer.
        
        Args:
            text: Input text to analyze
            target_label: Target class label to explain
            nsamples: Number of SHAP samples
            split_by: 'word' (default) to split on words, 'sentence' to split on sentences
                - For sentence-wise attribution, use split_by='sentence'.
                - For word-wise attribution, use split_by='word'.
        Returns:
            List of (word or sentence, attribution) tuples
        """
        print(f"\n--- Running SHAP Attribution for label: {target_label} (split_by={split_by}) ---")

        def predict_fn(texts, mode = 'prob'):
            """Wrapper for model predictions."""
            scores = []
            for text in texts:
                metrics = self.LLM_Handler.get_classification_metrics(text, target_label)
                if mode == 'prob':
                    prob = metrics.get('normalized_prob_given_label')
                    scores.append(float(prob))
                elif mode =='log':
                    log_prob = metrics.get('normalized_log_prob_given_label')
                    scores.append(float(log_prob))
            return np.array(scores).reshape(-1, 1)

        try:
            masker = shap.maskers.Text(self.get_split_function(split_by=split_by))
            explainer = shap.Explainer(
                predict_fn,
                masker,
                output_names=[target_label],
                max_evals=nsamples
            )
        
            shap_values = explainer([text])

            sources = shap_values.data[0]
            shap_values = shap_values.values[0]
            # print(f'shap sources:{sources}')
            # print(f'shap values:{shap_values}')

            # removes all trailing whitespace characters (spaces, tabs, etc.) from the end of the string.
            clean_sources = []
            for source in sources:
                clean = self.clean_word(source)
                if clean:  # Only include non-empty
                    clean_sources.append(clean)

            shap_coeffs = [float(s) for s in shap_values]
            return clean_sources, shap_coeffs 
            
        except Exception as e:
            print(f"Error in SHAP attribution: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    
    def compute_spearman_correlation(self, parts1, parts2, attrs1, attrs2):
        """
        Compute Spearman correlation between two attribution lists (floats), using the same partition.
        """
        if len(attrs1) == 0 or len(attrs2) == 0 or not np.array_equal(parts1, parts2):
            return 0.0
        return spearmanr(attrs1, attrs2)[0]
    
    def compute_log_prob_drop(self, text: str, attributions: List[Tuple[str, float]], k: int = 1) -> Dict[str, float]:
        """
        Compute the drop in log probability when removing top k most positively contributing words.
        
        Args:
            text: Original input text
            attributions: List of (word, score) tuples from attribution method
            k: Number of top words to remove (with positive effect)
        
        Returns:
            Dict containing metrics about the probability changes
        """
        original_label = self.LLM_Handler.get_predicted_class(text)
        original_metrics = self.LLM_Handler.get_classification_metrics(text, original_label)
        original_log_prob = original_metrics['normalized_log_prob_given_label']

        # Only consider words with positive attributions
        positive_attributions = [(word, score) for word, score in attributions if score > 0]
        # Sort by score descending (most positive effect first)
        sorted_positive = sorted(positive_attributions, key=lambda x: x[1], reverse=True)
        words_to_remove = [word for word, _ in sorted_positive[:k]]
        print(f'we should remove these words: {words_to_remove}')
        actual_k = len(words_to_remove)
        if actual_k < k:
            print(f"Warning: Only found {actual_k} sources with positive effect to remove (requested k={k}).")

        # Create modified text by replacing top k positive words with __
        modified_text = text
        for word in words_to_remove:
            modified_text = re.sub(r'\b' + re.escape(word) + r'\b', '__', modified_text)

        modified_label = self.LLM_Handler.get_predicted_class(modified_text)
        modified_metrics = self.LLM_Handler.get_classification_metrics(modified_text, original_label)
        modified_log_prob = modified_metrics['normalized_log_prob_given_label']

        return {
            'original_label': original_label,
            'original_log_prob': original_log_prob,
            'modified_log_prob': modified_log_prob,
            'log_prob_drop': original_log_prob - modified_log_prob,
            'modified_label': modified_label,
            'removed_words': words_to_remove,
            'num_positive_removed': actual_k,
            'requested_k': k
        }

    def clean_word(self, word):
        return word.rstrip()


    def get_split_function(self, split_by: str) -> Callable[[str, bool], Dict[str, Union[List[str], List[Tuple[int, int]]]]]:
        """
        Returns a splitter function that tokenizes text into words or sentences for SHAP's maskers.Text.
        For sentences: uses NLTK's sentence tokenizer.
        For words: uses NLTK's word_tokenize.
        
        Args:
            split_by (str): Tokenization mode, either 'word' or 'sentence'.
        
        Returns:
            Callable[[str, bool], Dict[str, Union[List[str], List[Tuple[int, int]]]]]:
                A function that takes a text string and a boolean (return_offsets_mapping).
                Returns a dict with 'input_ids' (list of tokens) and, if return_offsets_mapping=True,
                'offset_mapping' (list of (start, end) tuples).
        
        Raises:
            ValueError: If split_by is not 'word' or 'sentence'.
        """        
        if split_by == "sentence":
            def splitter(text: str, return_offsets_mapping: bool = True) -> Dict[str, Union[List[str], List[Tuple[int, int]]]]:
                tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
                spans = list(tokenizer.span_tokenize(text))
                input_ids = [text[start:end] for start, end in spans]
                out = {"input_ids": input_ids}
                if return_offsets_mapping:
                    out["offset_mapping"] = spans
                return out
        elif split_by == "word":
            def splitter(text: str, return_offsets_mapping: bool = True) -> Dict[str, Union[List[str], List[Tuple[int, int]]]]:
                nlp = English()
                doc = nlp(text)
                input_ids = [token.text for token in doc]
                offset_ranges = [(token.idx, token.idx + len(token)) for token in doc]
                out = {"input_ids": input_ids}
                if return_offsets_mapping:
                    out["offset_mapping"] = offset_ranges
                return out
        else:
            raise ValueError("split_by must be 'word' or 'sentence'")
        return splitter

    def run_local_benchmark(self, data, k_values=[1,2,3], nsamples=1000, split_by='word', save_dir='local_benchmark_results'):
        """
        Run a local benchmark over a list of texts, computing attribution metrics and plotting results.
        """
        
        os.makedirs(save_dir, exist_ok=True)
        methods = ['lasso', 'orthogonal', 'shap', 'cate']
        results = {method: {k: {'log_prob_drops': [], 'label_changes': []} for k in k_values} for method in methods}
        spearman_results = {('lasso', 'orthogonal'): [], ('lasso', 'shap'): [], ('orthogonal', 'shap'): [], ('cate', 'lasso'): [], ('cate', 'orthogonal'): [], ('cate', 'shap'): []}

        for text in data:
            target_label = self.LLM_Handler.get_predicted_class(text)
            # Compute attributions
            ame_output = self.attribute_ame(text, num_datasets=nsamples, split_by=split_by)
            parts = ame_output["partition"]
            lasso_coeffs = ame_output["lasso_coefficients"]
            ortho_coeffs = ame_output["ortho_coefficients"]
            cate_coeffs = ame_output["ortho_cate_attributions"]
            shap_parts, shap_coeffs = self.attribute_shap(text, target_label, nsamples=nsamples, split_by=split_by)
            ame_lasso_attributions = list(zip(parts, lasso_coeffs))
            ame_ortho_attributions = list(zip(parts, ortho_coeffs))
            shap_attributions = list(zip(shap_parts, shap_coeffs))
            cate_attributions = list(zip(parts, cate_coeffs))
            # For each method, for each k, compute log prob drop and label change
            for method, attributions in zip(methods, [ame_lasso_attributions, ame_ortho_attributions, shap_attributions, cate_attributions]):
                for k in k_values:
                    metrics = self.compute_log_prob_drop(text, attributions, k)
                    # If compute_log_prob_drop returns a float, treat as log_prob_drop only
                    if isinstance(metrics, dict):
                        log_prob_drop = metrics.get('log_prob_drop', 0)
                        label_change = metrics.get('original_label') != metrics.get('modified_label')
                    else:
                        log_prob_drop = metrics
                        # For backward compatibility, recompute label change
                        modified_text = text
                        sorted_words = sorted(attributions, key=lambda x: abs(x[1]), reverse=True)
                        words_to_remove = [word for word, _ in sorted_words[:k]]
                        for word in words_to_remove:
                            modified_text = re.sub(r'\\b' + re.escape(word) + r'\\b', '__', modified_text)
                        modified_label = self.LLM_Handler.get_predicted_class(modified_text)
                        label_change = (self.LLM_Handler.get_predicted_class(text) != modified_label)
                    results[method][k]['log_prob_drops'].append(log_prob_drop)
                    results[method][k]['label_changes'].append(label_change)

            
            spearman_results[('lasso', 'orthogonal')].append(self.compute_spearman_correlation(parts1=parts, parts2=parts, attrs1=lasso_coeffs, attrs2=ortho_coeffs))
            spearman_results[('lasso', 'shap')].append(self.compute_spearman_correlation(parts1=parts, parts2=shap_parts, attrs1=lasso_coeffs, attrs2=shap_coeffs))
            spearman_results[('orthogonal', 'shap')].append(self.compute_spearman_correlation(parts1=parts, parts2=shap_parts, attrs1=ortho_coeffs, attrs2=shap_coeffs))
            spearman_results[('cate', 'lasso')].append(self.compute_spearman_correlation(parts1=parts, parts2=shap_parts, attrs1=cate_coeffs, attrs2=lasso_coeffs))
            spearman_results[('cate', 'orthogonal')].append(self.compute_spearman_correlation(parts1=parts, parts2=shap_parts, attrs1=cate_coeffs, attrs2=ortho_coeffs))
            spearman_results[('cate', 'shap')].append(self.compute_spearman_correlation(parts1=parts, parts2=shap_parts, attrs1=cate_coeffs, attrs2=shap_coeffs))


        # Aggregate results
        summary = {method: {} for method in methods}
        for method in methods:
            for k in k_values:
                drops = results[method][k]['log_prob_drops']
                changes = results[method][k]['label_changes']
                summary[method][k] = {
                    'mean_log_prob_drop': np.mean(drops),
                    'std_log_prob_drop': np.std(drops),
                    'label_change_rate': np.mean(changes),
                }
        mean_spearman = {pair: np.nanmean(vals) for pair, vals in spearman_results.items()}

        # Plot results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Log prob drop
        plt.figure(figsize=(10,6))
        width = 0.25
        x = np.arange(len(k_values))
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f1c40f']
        for i, (method, color) in enumerate(zip(methods, colors)):
            means = [summary[method][k]['mean_log_prob_drop'] for k in k_values]
            stds = [summary[method][k]['std_log_prob_drop'] for k in k_values]
            plt.bar(x + i*width, means, width, label=method.upper(), color=color, yerr=stds, capsize=5)
        plt.xlabel('Top-k Words Removed')
        plt.ylabel('Log Probability Drop')
        plt.title('Impact of Removing Top-k Words by Attribution Method')
        plt.xticks(x + width, [f'k={k}' for k in k_values])
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'log_prob_drops_{timestamp}.png'))
        plt.close()
        # Label change rate
        plt.figure(figsize=(10,6))
        for i, (method, color) in enumerate(zip(methods, colors)):
            rates = [summary[method][k]['label_change_rate'] for k in k_values]
            plt.bar(x + i*width, rates, width, label=method.upper(), color=color)
        plt.xlabel('Top-k Words Removed')
        plt.ylabel('Label Change Rate')
        plt.title('Label Change Rate by Attribution Method')
        plt.xticks(x + width, [f'k={k}' for k in k_values])
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'label_change_rate_{timestamp}.png'))
        plt.close()
        # Spearman correlation
        plt.figure(figsize=(8,5))
        pairs = list(mean_spearman.keys())
        vals = [mean_spearman[pair] for pair in pairs]
        plt.bar([f'{a[0]} vs {a[1]}' for a in pairs], vals, color='#8e44ad')
        plt.ylabel('Mean Spearman Correlation')
        plt.title('Mean Spearman Correlation Between Attribution Methods')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'spearman_corr_{timestamp}.png'))
        plt.close()
        # Save summary stats
        with open(os.path.join(save_dir, f'summary_stats_{timestamp}.txt'), 'w') as f:
            f.write(f'Local Attribution Benchmark Results - {timestamp}\n')
            f.write('='*50 + '\n\n')
            for method in methods:
                f.write(f'{method.upper()} Results:\n')
                for k in k_values:
                    f.write(f'  Top-{k}: Log Prob Drop: {summary[method][k]["mean_log_prob_drop"]:.3f} ± {summary[method][k]["std_log_prob_drop"]:.3f}, Label Change Rate: {summary[method][k]["label_change_rate"]:.3f}\n')
                f.write('\n')
            f.write('Mean Spearman Correlations:\n')
            for pair, val in mean_spearman.items():
                f.write(f'  {pair[0]} vs {pair[1]}: {val:.3f}\n')
        print(f"Results saved in {save_dir} (timestamp: {timestamp})")



if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.2-1B"
    # model_name = "microsoft/Phi-3-mini-128k-instruct"
    # model_name = "meta-llama/Meta-Llama-3-8B"
    load_dotenv('/home/hrahmani/CausalContextAttributer/config.env')
    hf_auth_token  = os.environ.get("HF_API_KEY")
    
    class_labels = ["negative", "positive", "neutral"]
    llm_handler = LLM_Handler(class_labels, hf_auth_token, model_name)
    attributer = Attributer(llm_handler=llm_handler)

    # text = "i went and saw this movie last night after being coaxed to by a few friends of mine ." \
    # " I'll admit that i was reluctant to see it because from what i knew of ashton kutcher he was only able to do comedy . " \
    # "i was wrong . " \
    # "kutcher played the character of jake fischer very well , and kevin costner played ben randall with such professionalism ." \
    # " the sign of a good movie is that it can toy with our emotions . " \
    # "this one did exactly that . the entire theater ( which was sold out ) was overcome by laughter during the"
    # split_by = 'sentence'
    data = [
        "it's not good feeling.", "it's a negative sentiment", "I feel terrible, negative.", "I was wrong", "he doesn't like dark comedy", "An American came to post-World War II Europe and finds himself entangled in a dangerous mystery", "This booldy decision had great result", "he acted childish and awful."
        "Wow, this movie was an absolute masterpiece… if you enjoy wasting two hours on predictable clichés and terrible acting.", "The food was delicious, but the service was so slow I nearly fell asleep waiting.", "This phone isn't bad for the price, but it's not exactly a game-changer either.", "I'm sure the team worked so hard to deliver this buggy, overpriced software.", "The plot was thin, but the visuals were a feast for the eyes."
        "The concert was interesting, with unexpected twists in the performance.", "Oh, fantastic, another meeting that could've been an email.", "The hotel room was cozy, but the noise from the street was unbearable.", "The new policy ensures everyone gets a fair share, unless you're late to the meeting.", "The algorithm's precision was decent, but its recall left much to be desired."
        "This app isn't terrible, but it's hardly a must-have either.", "The performance wasn't amazing, but it got the job done.", "Not like this restaurant's food is anything to write home about.", "I can't say I'm thrilled with the results, but I'm not disappointed either.", "The service wouldn't be bad if they didn't ignore us for an hour."
    ]
    
    split_by = 'word'
    attributer.run_local_benchmark(data, k_values=[1,2,3], nsamples=1000, split_by=split_by)

    # text = data[0]
    # target_label = llm_handler.get_predicted_class(text)
    # print(f'text:{text}  /////predicted label: {target_label}')
    # ame_output = attributer.attribute_ame(text, num_datasets=1000, split_by=split_by)
    # partition = ame_output["partition"]
    # lasso_coeffs = ame_output["lasso_coefficients"]
    # ortho_coeffs = ame_output["ortho_coefficients"]
    # cate_coeffs = ame_output["ortho_cate_attributions"]
    # lasso_attributions = list(zip(partition, lasso_coeffs))
    # ortho_attributions = list(zip(partition, ortho_coeffs))
    # cate_attributions = list(zip(partition, cate_coeffs))
    # print("\nAME lasso Attributions:")
    # for word, score in lasso_attributions:
    #     print(f"{word}: {float(score):.4f}")
    # print("by AME lasso normalized log probability drops by", attributer.compute_log_prob_drop(text, lasso_attributions, k=1))

    # print("\nAME orthogonal Attributions:")
    # for word, score in ortho_attributions:
    #     print(f"{word}: {float(score):.4f}")
    # print("by AME Orthogonal normalized log probability drops by", attributer.compute_log_prob_drop(text, ortho_attributions, k=1))

    # print(f'the spaersman correlation between lasso and orthogonal {attributer.compute_spearman_correlation(parts1=partition, parts2=partition, attrs1=lasso_coeffs, attrs2=ortho_coeffs)}')

    # print("\nAME cate Attributions:")
    # for word, score in cate_attributions:
    #     print(f"{word}: {float(score):.4f}")
    # print("by AME cate normalized log probability drops by", attributer.compute_log_prob_drop(text, cate_attributions, k=1))

    # print(f'the spaersman correlation between cate and orthogonal {attributer.compute_spearman_correlation(parts1=partition, parts2=partition, attrs1=cate_coeffs, attrs2=ortho_coeffs)} and between cate and lasso {attributer.compute_spearman_correlation(parts1=partition, parts2=partition, attrs1=lasso_coeffs, attrs2=cate_coeffs)}')



    # # SHAP attribution and alignment
    # shap_parts, shap_coeffs = attributer.attribute_shap(text, target_label, nsamples=1000, split_by=split_by)
    # shap_coeffs = [float(s) for s in shap_coeffs]
    # shap_attributions = list(zip(shap_parts, shap_coeffs))
    # print("\nSHAP Attributions:")
    # for word, score in shap_attributions:
    #     print(f"{word}: {float(score):.4f}")
    # print("by SHAP normalized log probability drops by", attributer.compute_log_prob_drop(text, shap_attributions, k=1))

    # print(f'the spaersman correlation between shap and ame_lasso {attributer.compute_spearman_correlation(parts1=shap_parts, parts2=partition, attrs1=shap_coeffs, attrs2=lasso_coeffs)}')
    # print(f'the spaersman correlation between shap and ame_ortho {attributer.compute_spearman_correlation(parts1=shap_parts, parts2=partition, attrs1=shap_coeffs, attrs2=ortho_coeffs)}')
    # print(f'the spaersman correlation between shap and ame_cate {attributer.compute_spearman_correlation(parts1=shap_parts, parts2=partition, attrs1=shap_coeffs, attrs2=cate_coeffs)}')


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