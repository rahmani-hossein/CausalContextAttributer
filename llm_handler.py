# Attribution and Metrics Computation
import nltk
import math
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple
from dotenv import load_dotenv
import re
import os
from utils import split_text

class LLM_Handler:
    def __init__ (self, class_labels : List, hf_auth_token, model_name: str ):
       self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_auth_token)
       self.model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_auth_token).eval()
       self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       self.model.to(self.device)

       self.class_labels = class_labels
       self.class_tokens = {label: self.tokenizer.encode(label, add_special_tokens=False)[0]
                          for label in self.class_labels}
       
       if torch.cuda.is_available():
           allocated = torch.cuda.memory_allocated(self.device) / 1024**3  # in GB
           max_allocated = torch.cuda.max_memory_allocated(self.device) / 1024**3  # in GB
           print(f"Current GPU memory allocated: {allocated:.2f} GB")
           print(f"Max GPU memory allocated: {max_allocated:.2f} GB")

  
    def get_classification_metrics(self, text: str, label: str) -> Dict[str, float]:
        """
        Compute classification metrics for a sampled prompt (prompt with some words deleted) and label.

        Args:
            prompt (str): The input prompt.
            label (str): The label to compute log probability for.

        Returns:
            dict: A dictionary containing:
                - 'predicted_label': The predicted category (str).
                - 'log_prob_predicted': Log probability of the predicted category (float).
                - 'log_odds_predicted': Log odds of the predicted category (float).
                - 'log_prob_given_label': Log probability of the given label (float).
        """
        # Encode the prompt
        prompt = f"Classify this headline: {text}. The category is: "
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :]  # Logits for the next token
            probs = torch.softmax(logits, dim=-1)

        # Extract log probabilities for all class labels
        class_probs = {lbl: probs[0, token].item() for lbl, token in self.class_tokens.items()}

        predicted_label = max(class_probs, key=class_probs.get)
        # log_prob_predicted = class_log_probs[predicted_label]

        # Get log probability of the given label
        prob_given_label = class_probs[label]
        log_prob_given_label = math.log(prob_given_label) if prob_given_label > 0 else float('-inf')

        if prob_given_label < 1:
            log_odds_given_label = math.log(prob_given_label/ (1-prob_given_label))
        else:
            log_odds_given_label = float('inf')  # Handle P=1 case
        
        # Calculate sum of probabilities of all class labels
        sum_class_probs = sum(class_probs.values())

        # Compute normalized probability for given label
        if sum_class_probs > 0:
            normalized_prob_given_label = prob_given_label / sum_class_probs
            normalized_prob_predicted_label = class_probs[predicted_label] / sum_class_probs

        else:
            normalized_prob_given_label = 0.0
            normalized_prob_predicted_label = 0.0
        

        # Return all metrics in a dictionary
        return {
            'prob_given_label':prob_given_label,
            'predicted_label': predicted_label,
            # 'log_odds_given_label': log_odds_given_label,
            'log_prob_given_label': log_prob_given_label,
            'prob_predicted_label': class_probs[predicted_label],
            'normalized_prob_given_label': normalized_prob_given_label,
            'normalized_log_prob_given_label': math.log(normalized_prob_given_label) if normalized_prob_given_label > 0 else float('-inf'),
            'normalized_log_prob_predicted_label': math.log(normalized_prob_predicted_label) if normalized_prob_predicted_label > 0 else float('-inf')
        }
    

    def get_predicted_class(self, headline: str) -> str:
        """Get the predicted class for a headline."""
        prompt = f"Classify this headline: {headline}. The category is: "
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            class_probs = {label: probs[0, token].item() for label, token in self.class_tokens.items()}
            return max(class_probs, key=class_probs.get)




# # Example Usage
# if __name__ == "__main__":
#     model_name="meta-llama/Llama-3.2-1B"
#     load_dotenv('/home/hrahmani/CausalContextAttributer/config.env')
#     hf_auth_token  = os.environ.get("HF_API_KEY")
#     class_labels = ["Technology", "Politics", "Sports", "Art", "Other"]
#     LLM_Handler = LLM_Handler(class_labels, hf_auth_token, model_name)

#     # Single data point
#     headline = "this painting from sport team barcelona is amazing"
#     true_label = "Art"
#     import time
#     start_time = time.time()
#     res = LLM_Handler.get_classification_metrics(headline, true_label)
#     end_time = time.time()
#     time_taken = end_time - start_time
#     print(f"Time taken for llm call: {time_taken} seconds")
#     print(res)

class LlamaHandler:
    """
    Handler for Llama 3 8B-Instruct/phi mini for both classification (one-word) and question answering (one-sentence) tasks.
    Provides log probability of the generated output.
    """
    def __init__(self, class_labels, hf_auth_token, model_name):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_auth_token)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_auth_token,
                                                          torch_dtype=torch.bfloat16).eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"LlamaHandler: Model '{model_name}' has {sum(p.numel() for p in self.model.parameters()):,} parameters.")
        self.model.to(self.device)
        self.class_labels = class_labels

    def classify(self, text, true_label):
        """
        Classifies the input text into one of the class labels, returns (label, logprob).
        """
        label_list = ", ".join(f'"{lbl}"' for lbl in self.class_labels)
        messages = [
            {"role": "system", "content": f"You are a helpful assistant that classifies text sentiment. Only answer with one word, which must be one of: {label_list}."},
            {"role": "user", "content": f'Classify the following text into one of the categories ({label_list}). Only output the category word, nothing else.\\n\\nText: "{text}"'},
        ]
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=2,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False
            )
        generated_ids = outputs.sequences[0][inputs.shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        print(f'the generated text is {generated_text}')
        # Find the best matching label
        for label in self.class_labels:
            if re.fullmatch(label, generated_text, re.IGNORECASE):
                chosen_label = label
                break
        else:
            print(f'we fall for finding label for {text}')
            chosen_label = generated_text.split()[0]  # fallback: first word
        # Compute log probability of the generated label token
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        logprob = transition_scores[0][0].item() if transition_scores[0].numel() > 0 else float('-inf')
        return chosen_label, logprob

    def answer_one_sentence(self, question, context=None):
        """
        Answers the question in exactly one sentence, returns (sentence, logprob).
        If context is provided, it is included in the prompt.
        """
        if context:
            user_content = f"Answer the following question in exactly one sentence.\\n\\nContext: {context}\\n\\nQuestion: {question}"
        else:
            user_content = f"Answer the following question in exactly one sentence.\\n\\nQuestion: {question}"
        messages = [
            {"role": "system", "content": "You are a helpful assistant that always answers in exactly one sentence."},
            {"role": "user", "content": user_content},
        ]
        inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=64,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False
            )
        generated_ids = outputs.sequences[0][inputs.shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        # Extract only the first sentence
        sentence = re.split(r'(?<=[.!?]) +', generated_text)[0]
        # Compute log probability of the generated sentence tokens
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        logprob = transition_scores[0][:len(generated_ids)].sum().item() if transition_scores[0].numel() > 0 else float('-inf')
        return sentence, logprob

# Example usage for testing
if __name__ == "__main__":
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    from dotenv import load_dotenv
    load_dotenv('/home/hrahmani/CausalContextAttributer/config.env')
    hf_auth_token  = os.environ.get("HF_API_KEY")
    class_labels = ["positive", "negative", "neutral"]
    handler = LlamaHandler(class_labels, hf_auth_token, model_name)
    text = "Oh, fantastic, another meeting that could've been an email."
    true_label = "negative"
    import time
    start_time = time.time()
    metrics = handler.classify(text, true_label)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(metrics)