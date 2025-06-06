import nltk
import numpy as np
import re
from typing import List, Tuple, Dict
from spacy.lang.en import English
from nltk.tokenize import TweetTokenizer
from dataclasses import dataclass
from functools import lru_cache
from llm_handler import LLM_Handler
from spacy.lang.en import English

@dataclass
class TextPartition:
    """
    Class to hold partitioned text information.
    
    parts: the individual text fragments (sentences or words).
    indices: list of (start, end) character offsets for each part in the original text.
    part_type: either 'sentence' or 'word' to indicate partition type.
    """
    parts: List[str]
    indices: List[Tuple[int, int]]
    part_type: str


class EfficientTextProcessor:
    """
    Efficient text processor that:
     - Splits text into sentences or words, preserving start/end indices.
     - Creates masked samples (where some parts are replaced by '__') 
       with probability p of keeping each part.
     - Caches partitioning results to speed up repeated usage.
    """
    
    def __init__(self, pval: np.ndarray = np.array([0.2, 0.4, 0.6, 0.8])):
        """
        Args:
            pval: array of possible probabilities to pick from when masking.
        """
        self.pval = pval
        
        # Lazy-loaded tokenizers
        self.sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.spacy_nlp = English()
        # Pre-compile a word pattern for regex-based splitting
        self._word_pattern = re.compile(r'\b\w+\b')


    def split_text(self, text: str, split_by: str = "sentence") -> TextPartition:
        """
        Split `text` into parts (sentences or words), track their character offsets, and cache results.
        
        Args:
            text: The original text to be split.
            split_by: Either 'sentence' or 'word'.
        
        Returns:
            A TextPartition object containing the parts, their indices, and the partition type.
        """

        if split_by == "sentence":
            parts = []
            indices = []
            for span in self.sentence_tokenizer.span_tokenize(text):
                start, end = span
                parts.append(text[start:end])
                indices.append((start, end))
        elif split_by == "word":
            doc = self.spacy_nlp(text)
            parts = [token.text for token in doc]
            indices = [(token.idx, token.idx + len(token)) for token in doc]
        else:
            raise ValueError(f"Invalid split_by value: {split_by}")

        partition = TextPartition(parts=parts, indices=indices, part_type=split_by)

        return partition
    
    
    # def split_text(self, text: str, split_by: str = "sentence") -> TextPartition:
    #     """
    #     Split `text` into parts (sentences or words), track their character offsets, and cache results.
        
    #     Args:
    #         text: The original text to be split.
    #         split_by: Either 'sentence' or 'word'.
        
    #     Returns:
    #         A TextPartition object containing the parts, their indices, and the partition type.
    #     """
    #     cache_key = self._get_partition_key(text, split_by)
    #     if cache_key in self._partition_cache:
    #         return self._partition_cache[cache_key]

    #     if split_by == "sentence":
    #         parts = []
    #         indices = []
    #         # Use NLTK span_tokenize to get sentences plus their offsets
    #         for span in self.sentence_tokenizer.span_tokenize(text):
    #             start, end = span
    #             parts.append(text[start:end])
    #             indices.append((start, end))
                
    #     elif split_by == "word":
    #         parts = []
    #         indices = []
    #         # Use regex or spaCy to find words plus their offsets
    #         # Here we go with the regex approach for efficiency
    #         for match in self._word_pattern.finditer(text):
    #             start, end = match.span()
    #             parts.append(text[start:end])
    #             indices.append((start, end))
    #     else:
    #         raise ValueError(f"Invalid split_by value: {split_by}")

    #     partition = TextPartition(parts=parts, indices=indices, part_type=split_by)
    #     self._partition_cache[cache_key] = partition
    #     return partition


    def create_masked_sample(self, 
                         text: str, 
                         partition: TextPartition, 
                         p: float) -> Tuple[str, np.ndarray]:
        """
        Create a masked sample by replacing some parts with '__'.
        
        Args:
            text: The original full text.
            partition: The partition info (parts + indices).
            p: Probability to keep each part.
        
        Returns:
            A tuple of (masked_text: str, mask: np.ndarray), where mask[i] is True if part i is kept, False if masked.
        """
        mask = np.random.random(len(partition.parts)) < p
        result_parts = []
        last_end = 0
        for (start, end), keep in zip(partition.indices, mask):
            if start > last_end:
                result_parts.append(text[last_end:start])
            if keep:
                result_parts.append(text[start:end])
            else:
                result_parts.append("__")
            last_end = end
        if last_end < len(text):
            result_parts.append(text[last_end:])

        masked_text = "".join(result_parts)
        return masked_text, mask


    def generate_samples(self, 
                        text: str, 
                        num_samples: int = 5, 
                        split_by: str = "sentence") -> List[str]:
        """
        Experimental usecase. In general we need probabilites and the standard way of using it to call masked function in prompt generator class. 
        Generate multiple masked samples from the text.
        Each sample chooses a random probability p from self.pval.
        
        Args:
            text: The full text to partition and mask.
            num_samples: How many masked samples to generate.
            split_by: 'sentence' or 'word'.
        
        Returns:
            A list of masked text strings.
        """
        partition = self.split_text(text, split_by)
        
        # Choose random probabilities for each sample (drawn from pval)
        probabilities = np.random.choice(self.pval, size=num_samples)
        
        # Create each sample
        samples = []
        for p in probabilities:
            masked, mask = self.create_masked_sample(text, partition, p)
            samples.append(masked)
        
        return samples

class EfficientPromptGenerator:
    """
    Example usage class: Takes an initial prompt, uses an EfficientTextProcessor
    to generate masked samples, and can also produce a simple feature matrix X.
    """
    def __init__(self, 
                 LLM_handler: LLM_Handler,
                 prompt: str, 
                 num_datasets: int = 5,
                 pval: np.ndarray = np.array([0.2, 0.4, 0.6, 0.8])):
        """
        Args:
            prompt: The original prompt text.
            num_datasets: Number of masked samples (datasets) to create.
            pval: Array of probabilities from which we draw p for masking.
        """
        self.original_prompt = prompt
        self.num_datasets = num_datasets
        self.pval = pval
        self.LLM_Handler = LLM_handler
        self.text_processor = EfficientTextProcessor(pval=pval)
        
        # Will hold the actual masked samples once created
        self.sample_prompts: List[str] = []
        self.ps: np.ndarray = None  # Store the probabilities used for each sample

    def coef_scaling(self) -> float:
        return (1/(self.pval *(1-self.pval))).mean()

    def sample_pvals_reweighted(self, p_candidates: np.ndarray, size: int) -> np.ndarray:
        """
        Draw 'size' many p-values from p_candidates according to the
        1 / [p(1-p)] reweighting rule (then normalize the weights) fpr p-feaurization setting.
        """
        weights = 1.0 / (p_candidates * (1.0 - p_candidates))   # 1/[p(1-p)] for each candidate
        weights /= weights.sum()                                # normalize so they sum to 1
        return np.random.choice(p_candidates, size=size, p=weights)

    def create_X(self, mode = "1/p featurization", split_by: str = "sentence") -> np.ndarray:
        """
        Build a matrix X where each row is a masked sample, and each column is a part.
        
        Args:
            mode: "1/p featurization" or "p featurization".
            split_by: 'sentence' or 'word'.
        
        Returns:
            A tuple (X: np.ndarray, partition: TextPartition).
        """
        if split_by == "sentence":
            partition = self.text_processor.split_text(self.original_prompt, "sentence")
        else:
            partition = self.text_processor.split_text(self.original_prompt, "word")
        
        # Draw the probabilities for each of the masked samples
        if mode == "1/p featurization":
            self.ps = np.random.choice(self.pval, size=self.num_datasets)
        elif mode == "p featurization":
            self.ps = self.sample_pvals_reweighted(self.pval, size = self.num_datasets)
        
        # Generate samples and collect masks
        self.sample_prompts = []
        self.masks = []
        for p in self.ps:
            masked_text, mask = self.text_processor.create_masked_sample(self.original_prompt, partition, p)
            # print(f'for p: {p} we created this {masked_text} with this mask: {mask}')
            self.sample_prompts.append(masked_text)
            self.masks.append(mask)
        
        N = len(partition.parts)
        X = np.zeros((self.num_datasets, N))
        nu = self.coef_scaling()
        for m, (p, mask) in enumerate(zip(self.ps, self.masks)):
            for i, keep in enumerate(mask):
                if keep:
                    if mode == "1/p featurization":
                        X[m, i] = 1.0 / (p * np.sqrt(nu))
                    elif mode == "p featurization":
                        X[m, i] = np.sqrt((1 - p) / nu)
                else:  # masked
                    if mode == "1/p featurization":
                        X[m, i] = -1.0 / ((1 - p) * np.sqrt(nu))
                    elif mode == "p featurization":
                        X[m, i] = -np.sqrt(p / nu)
            # print(f'values for {m} th sample: {X[m,:]}')

        return X, partition
    
    def create_y(self, sampled_prompts, original_label):


        y = np.zeros(len(sampled_prompts))
        y_lognormalized = np.zeros(len(sampled_prompts))
        predicted_label = []
        for i in range(len(sampled_prompts)):
            llm_res = self.LLM_Handler.get_classification_metrics(sampled_prompts[i], original_label)
            y[i] = llm_res['log_prob_given_label']
            
            y_lognormalized[i] = llm_res['normalized_log_prob_given_label']
            predicted_label.append(llm_res['predicted_label'])


        return y, y_lognormalized, predicted_label
        


# --- Quick Test / Example ---
if __name__ == "__main__":
    # import nltk
    # nltk.download('punkt_tab')

    # Example usage text
    text = (
        "Hello world! This is a test. "
        "We will see how sentence partitioning works. "
        "Then we try word partitioning."
    )

    # Create the main text processor
    processor = EfficientTextProcessor()

    # Get masked samples (sentence-level)
    masked_samples = processor.generate_samples(text, num_samples=3, split_by="sentence")
    print("\n--- Masked Samples (Sentence) ---")
    for s in masked_samples:
        print(s, "\n")

    # Get masked samples (word-level)
    masked_samples = processor.generate_samples(text, num_samples=3, split_by="word")
    print("\n--- Masked Samples (Word) ---")
    for s in masked_samples:
        print(s, "\n")

    # Prompt generator usage
    prompt_gen = EfficientPromptGenerator(prompt=text, num_datasets=3)
    X, partition = prompt_gen.create_X(split_by="sentence", mode= "p featurization")
    print("\n--- Feature Matrix X (Sentence) ---")
    print(X)