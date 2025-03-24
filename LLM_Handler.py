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
       load_dotenv('config.env')
       hf_auth_token  = os.environ.get("HF_API_KEY")
       self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_auth_token)
       self.model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_auth_token).eval()
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
            print(probs.shape)
            return max(class_probs, key=class_probs.get)