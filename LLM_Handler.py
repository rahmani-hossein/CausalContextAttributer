# Attribution and Metrics Computation
import nltk
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple
from dotenv import load_dotenv
import re
import os

class LLM_Handler:
  def __init__ (self, class_labels : List, model_name: str = "meta-llama/Llama-3.2-1B"):
       load_dotenv('/home/hrahmani/CausalContextAttributer/config.env')
       hf_auth_token  = os.environ.get("HF_API_KEY")
      #  print("authentication token: ",  hf_auth_token)
       self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_auth_token)
       self.model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_auth_token).eval()
       self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       self.model.to(self.device)

       self.class_labels = class_labels
       self.class_tokens = {label: self.tokenizer.encode(label, add_special_tokens=False)[0]
                          for label in self.class_labels}

  def getClassification_log_prob(self, prompt: str, label: str) -> float:
        """Compute log probability of a label given a sampled prompt (prompt with some words deleted)."""

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :]  # Logits for next token
            log_probs = torch.log_softmax(logits, dim=-1)
            token_id = self.class_tokens[label]
            return log_probs[0, token_id].item()

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
        

        
class AttributionCalculator:
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B"):  # Use an accessible model
        """Initialize the model and tokenizer."""
        class_labels = ["Technology", "Politics", "Sports", "Art", "Other"]
        self.LLM_Handler = LLM_Handler(class_labels, model_name)



# Example Usage
if __name__ == "__main__":
    # Note: Llama 3-8B may require special access; using Llama 2-7B as a placeholder     meta-llama/Meta-Llama-3-8B
    calc = AttributionCalculator(model_name="meta-llama/Llama-3.2-1B")  # Adjust as needed

    # Single data point
    headline = "Election Results Announced Today"
    true_label = "Politics"
    import time
    start_time = time.time()
    label = calc.LLM_Handler.get_predicted_class(headline)
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Time taken for llm call: {time_taken} seconds")
    print(label)

    print(calc.LLM_Handler.getClassification_log_prob("__ Results Announced Today", label))


    headline2 = " __ Results Announced Today"
    true_label = "Politics"
    label2 = calc.LLM_Handler.get_predicted_class(headline2)
    print(label2)
    print(calc.LLM_Handler.getClassification_log_prob(headline2, label2))
    # Dataset
    dataset = [
        ("Election Results Announced Today", "politics"),
        ("Football Match Ends in Draw", "sports"),
        ("New Movie Released This Week", "entertainment")
    ]
    # metrics = calc.evaluate_dataset(dataset, max_k=3)
    # print("\nDataset Metrics:")
    # for metric, value in metrics.items():
    #     print(f"{metric}: {value}")
