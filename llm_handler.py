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

class LLM_Handler:
    def __init__ (self, class_labels : List, hf_auth_token, model_name: str ):
       self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_auth_token)
       self.model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_auth_token).eval()
       self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       self.model.to(self.device)

       self.class_labels = class_labels
       self.class_tokens = {label: self.tokenizer.encode(label, add_special_tokens=False)[0]
                          for label in self.class_labels}
       

  
    def get_classification_metrics(self, prompt: str, label: str) -> Dict[str, float]:
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
        print("we get the prompt for the LLM: ", prompt)
        prompt = f"Classify this headline: {prompt}. The category is: "
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


from dotenv import load_dotenv
import os
# Example Usage
if __name__ == "__main__":
    model_name="meta-llama/Llama-3.2-1B"
    load_dotenv('/home/hrahmani/CausalContextAttributer/config.env')
    hf_auth_token  = os.environ.get("HF_API_KEY")
    class_labels = ["Technology", "Politics", "Sports", "Art", "Other"]
    LLM_Handler = LLM_Handler(class_labels, hf_auth_token, model_name)

    # Single data point
    headline = "this painting from sport team barcelona is amazing"
    true_label = "Art"
    import time
    start_time = time.time()
    res = LLM_Handler.get_classification_metrics(headline, true_label)
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Time taken for llm call: {time_taken} seconds")
    print(res)

    # print(calc.LLM_Handler.getClassification_log_prob("__ Results Announced Today", label))


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
