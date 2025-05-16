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
import nltk
from spacy.lang.en import English
from typing import Callable, List, Tuple, Dict, Union



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

        # Define prediction function
        def predict_fn(texts, mode = 'prob'):
            """Wrapper for model predictions."""
            print(texts)
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
        
        # Use regex for SHAP splitting
        if split_by == 'word':
            masker_regex = r"\W"  # split on non-words (word-level)
        elif split_by == 'sentence':
            masker_regex = r"(?<=[.!?])\s+|(?<=[.!?])$"  # split on sentence boundaries
        else:
            raise ValueError(f"split_by must be 'word' or 'sentence', got {split_by}")

        try:
            masker = shap.maskers.Text(masker_regex)
            explainer = shap.Explainer(
                predict_fn,
                masker,
                output_names=[target_label],
                max_evals=nsamples
            )
            
            # Compute SHAP values
            shap_values = explainer([text])

            sources = shap_values.data[0]
            attributions = shap_values.values[0]
            
            # Match sources (words/sentences) with their attributions
            word_attributions = []
            for source, attr in zip(sources, attributions):
                clean = self.clean_word(source)
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
        shap_scores = {source: score for source, score in shap_attrs}
        ame_scores = {source: score for source, score in ame_attrs}
        
        # Get common words
        common_words = set(shap_scores.keys()) & set(ame_scores.keys())
        print(f'common words are: {common_words}')
        
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
        # nltk.download('punkt', quiet=True)
        # nltk.download('punkt_tab', quiet=True)
        
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
                input_ids = nltk.word_tokenize(text)
                offset_ranges = []
                pos = 0
                for token in input_ids:
                    start = text.find(token, pos)
                    if start == -1:  # Handle rare cases where token isn't found
                        start = pos
                    end = start + len(token)
                    offset_ranges.append((start, end))
                    pos = end
                out = {"input_ids": input_ids}
                if return_offsets_mapping:
                    out["offset_mapping"] = offset_ranges
                # Optional: Filter punctuation to match regex r'\b\w+\b' behavior
                # if not return_offsets_mapping:
                #     out["input_ids"] = [t for t in input_ids if re.match(r'\w+', t)]
                return out
        else:
            raise ValueError("split_by must be 'word' or 'sentence'")
        return splitter



    # def get_split_function(self, split_by: str):
    #     """
    #     Returns a splitter function that tokenizes text into words or sentences.
    #     For sentences: uses NLTK's sentence tokenizer.
    #     For words: uses regex pattern r'\b\w+\b' to match AME's approach.
        
    #     Args:
    #         split_by (str): Tokenization mode, either 'word' or 'sentence'.
        
    #     Returns:
    #         Callable[[str, bool], Union[List[str], Dict[str, Union[List[str], List[Tuple[int, int]]]]]]:
    #             A function that takes a text string and a boolean (return_offsets_mapping).
    #             - If return_offsets_mapping=False, returns a list of tokens (for AME).
    #             - If return_offsets_mapping=True, returns a dict with 'input_ids' (list of tokens)
    #             and 'offset_mapping' (list of (start, end) tuples) for SHAP.
        
    #     Raises:
    #         ValueError: If split_by is not 'word' or 'sentence'.
    #     """    
    #     if split_by == "sentence":
    #         def splitter(text: str, return_offsets_mapping: bool = True):
    #             tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    #             spans = list(tokenizer.span_tokenize(text))
    #             input_ids = [text[start:end] for start, end in spans]
    #             out = {"input_ids": input_ids}
    #             if return_offsets_mapping:
    #                 out["offset_mapping"] = spans
    #             return out
    #     elif split_by == "word":
    #         def splitter(text: str, return_offsets_mapping: bool = True):
    #             pattern = re.compile(r'\b\w+\b')
    #             input_ids = []
    #             offset_ranges = []
    #             for match in pattern.finditer(text):
    #                 start, end = match.span()
    #                 input_ids.append(text[start:end])
    #                 offset_ranges.append((start, end))
    #             out = {"input_ids": input_ids}
    #             if return_offsets_mapping:
    #                 out["offset_mapping"] = offset_ranges
    #             return out
    #     else:
    #         raise ValueError("split_by must be 'word' or 'sentence'")
    #     return splitter

    
    # Define prediction function
    def predict_fn(self, texts, mode = 'prob'):
        """Wrapper for model predictions."""
        print(texts)
        target_label = "negative"
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


def nltk_sentence_splitter(s, return_offsets_mapping=True):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    spans = list(tokenizer.span_tokenize(s))
    input_ids = [s[start:end] for start, end in spans]
    out = {"input_ids": input_ids}
    if return_offsets_mapping:
        out["offset_mapping"] = spans
    return out



if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.2-1B"
    load_dotenv('/home/hrahmani/CausalContextAttributer/config.env')
    hf_auth_token  = os.environ.get("HF_API_KEY")
    
    class_labels = ["negative", "positive"]
    llm_handler = LLM_Handler(class_labels, hf_auth_token, model_name)
    attributer = Attributer(llm_handler=llm_handler)
    # SHAP setup
    split_by = "word"
    masker = shap.maskers.Text(attributer.get_split_function(split_by="word"))
    explainer = shap.Explainer(attributer.predict_fn, masker)
    text = "i was wrong."
    # text = "i went and saw this movie last night after being coaxed to by a few friends of mine ." \
    # " I'll admit that i was reluctant to see it because from what i knew of ashton kutcher he was only able to do comedy . " \
    # "i was wrong . " \
    # "kutcher played the character of jake fischer very well , and kevin costner played ben randall with such professionalism ." \
    # " the sign of a good movie is that it can toy with our emotions . " \
    # "this one did exactly that . the entire theater ( which was sold out ) was overcome by laughter during the"
    # text = "i went and saw this movie last night after being coaxed to by a few friends of mine. i was wrong. It was a terrific movie."
    shap_values = explainer([text])
    print(shap_values.data[0])
    print(shap_values.values[0])
    ame_lasso_attributions, ame_ortho_attributions = attributer.attribute_ame(text, num_datasets=1000, split_by=split_by)

    # Print results
    print("\nAME lasso Attributions:")
    for word, score in ame_lasso_attributions:
        print(f"{word}: {score:.4f}")





    # text = "i went and saw this movie last night after being coaxed to by a few friends of mine ." \
    # " I'll admit that i was reluctant to see it because from what i knew of ashton kutcher he was only able to do comedy . " \
    # "i was wrong . " \
    # "kutcher played the character of jake fischer very well , and kevin costner played ben randall with such professionalism ." \
    # " the sign of a good movie is that it can toy with our emotions . " \
    # "this one did exactly that . the entire theater ( which was sold out ) was overcome by laughter during the"
    # split_by = 'sentence'
    # target_label = llm_handler.get_predicted_class(text)
    # print(f'predicted label is {target_label}')

    # # class_labels = ["Technology", "Politics", "Sports", "Art", "Other"]
    # # llm_handler = LLM_Handler(class_labels, hf_auth_token, model_name)
    # # attributer = Attributer(llm_handler=llm_handler)
    # # text = "Local Mayor Launches Initiative to enhance urban public transport."
    # # target_label = "Politics"

    # ame_lasso_attributions, ame_ortho_attributions = attributer.attribute_ame(text, num_datasets=1000, split_by=split_by)

    # # Print results
    # print("\nAME lasso Attributions:")
    # for word, score in ame_lasso_attributions:
    #     print(f"{word}: {score:.4f}")
    
    # print("by AME lasso normalized log probability drops by", attributer.compute_log_prob_drop(text, ame_lasso_attributions, k=1))

    # print("\nAME orthogonal Attributions:")
    # for word, score in ame_ortho_attributions:
    #     print(f"{word}: {score:.4f}")
    
    # print("by AME Orthogonal normalized log probability drops by", attributer.compute_log_prob_drop(text, ame_ortho_attributions, k=1))

    # print(f'the spaersman correlation between lasso and orthogonal {attributer.compute_spearman_correlation(shap_attrs=ame_ortho_attributions, ame_attrs=ame_lasso_attributions)}')

    # shap_attributions = attributer.attribute_shap(text, target_label, nsamples=1000, split_by=split_by)
    # print("\nSHAP Attributions:")
    # for word, score in shap_attributions:
    #     print(f"{word}: {score:.4f}")

    # print(f'the spaersman correlation between shap and ame_lasso{attributer.compute_spearman_correlation(shap_attrs=shap_attributions, ame_attrs=ame_lasso_attributions)}')

    
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