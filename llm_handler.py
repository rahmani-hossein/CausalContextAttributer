# Attribution and Metrics Computation
import nltk
import math
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple
from dotenv import load_dotenv
import os
from utils import split_text
import torch.nn.functional as F

class LLM_Handler:
    def __init__ (self, class_labels : List, hf_auth_token, model_name: str ):
       self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_auth_token)
       self.model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_auth_token).eval()
       self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       self.model.to(self.device)

       self.class_labels = class_labels
       self.class_tokens = {label: self.tokenizer.encode(label, add_special_tokens=False)[0]
                          for label in self.class_labels}

  
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

    def classify_with_logprob(self, text: str):
        """
        Classifies the input text and returns the chosen label, the unnormalized logprobs (logits) for all class labels, and the normalized logprobs for all class labels.
        Prints the normalized probabilities for all class labels for debugging.
        Returns:
            tuple: (chosen_label, unnormalized_logprobs_dict, normalized_logprobs_dict)
        """
        label_list = ", ".join(f'"{lbl}"' for lbl in self.class_labels)
        messages = [
            {"role": "system", "content": f"You are a helpful assistant that classifies text. Only answer with one word, which must be one of: {label_list}."},
            {"role": "user", "content": f'Classify the following text into one of the categories ({label_list}). Only output the category word, nothing else.\n\nText: "{text}"'},
        ]
        input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.device)
        attention_mask = (input_ids != self.tokenizer.eos_token_id).long()
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=3,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.eos_token_id)
        seq = outputs.sequences[0]
        gen_ids = seq[input_ids.shape[-1]:]
        generated_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # Get token IDs for the first token of each label
        label_token_ids = {}
        for label in self.class_labels:
            tokens = self.tokenizer.encode(label, add_special_tokens=False)
            if tokens:
                label_token_ids[label] = tokens[0]  # array with one value

        logits = outputs.scores[0]  # Shape: [1, vocab_size]
        # Unnormalized logprobs (logits) for each label
        probs = F.softmax(logits, dim=-1)[0, list(label_token_ids.values())]
        print(probs)
        logprobs = torch.log(probs)
        # Normalized probabilities and logprobs
        label_logits = torch.tensor([logits[0, label_token_ids[label]].item() for label in self.class_labels]).to(self.device)
        label_probs = F.softmax(label_logits, dim=-1)
        label_logprobs = torch.log(label_probs)

        # print("Normalized class probabilities (first token):")
        # for i, label in enumerate(self.class_labels):
        #     print(f"  {label}: {label_probs[i].item():.4f}")

        # Find the chosen label (highest normalized probability)
        chosen_idx = torch.argmax(label_probs).item()
        chosen_label = self.class_labels[chosen_idx]

        unnormalized_logprobs = {label: logprobs[i].item() for i, label in enumerate(self.class_labels)}
        normalized_logprobs = {label: label_logprobs[i].item() for i, label in enumerate(self.class_labels)}
        return {'chosen_label':chosen_label, 'unnormalized_logprobs':unnormalized_logprobs, 'normalized_logprobs':normalized_logprobs}

    def generate_one_sentence_with_logprob(self, question: str, context: str = None, max_new_tokens: int = 50):
        """
        Answers the question in exactly one sentence, returns (sentence, logprob).
        """
        if context:
            user_content = f"Answer the following question in exactly one sentence.\\n\\nContext: {context}\\n\\nQuestion: {question}"
        else:
            user_content = f"Answer the following question in exactly one sentence.\\n\\nQuestion: {question}"
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that always answers in exactly one sentence."},
            {"role": "user", "content": user_content},
        ]
        input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.device)
        attention_mask = (input_ids != self.tokenizer.eos_token_id).long()
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.eos_token_id)
        # decoding
        seq = outputs.sequences[0]
        gen_ids = seq[input_ids.shape[-1] :]
        generated_text = self.tokenizer.decode(
                gen_ids, skip_special_tokens=True
            ).strip()
        
       
        # Compute the log-probability
        log_probs = 0.0
        for i, logits in enumerate(outputs.scores):
            token_id = gen_ids[i]
            prob = F.softmax(logits, dim=-1)[0, token_id]
            log_probs += torch.log(prob)
        
            
        return generated_text, log_probs.item(), gen_ids

    def compute_target_log_prob(self, question: str, target_answer: str, context: str = None) -> float:
        """
        Computes the log probability of a target_answer given a question and context.
        """
        if context:
            user_content = f"Answer the following question in exactly one sentence.\\n\\nContext: {context}\\n\\nQuestion: {question}"
        else:
            user_content = f"Answer the following question in exactly one sentence.\\n\\nQuestion: {question}"
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that always answers in exactly one sentence."},
            {"role": "user", "content": user_content},
        ]
        
        # Get the prompt as a string (not tokenized)
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
        # Tokenize prompt and target answer
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").to(self.device).input_ids
        target_ids = self.tokenizer(target_answer, add_special_tokens=False, return_tensors="pt").to(self.device).input_ids
        input_ids = torch.cat([prompt_ids, target_ids], dim=1)
        attention_mask = (input_ids != self.tokenizer.eos_token_id).long()
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask, labels=input_ids)
            logits = outputs.logits

        # Shift logits and labels for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        # Compute log-probs for all tokens
        log_probs_full = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs_full.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

        # Only keep log-probs for the target answer tokens
        # The target answer starts after the prompt, so slice accordingly
        target_token_log_probs = token_log_probs[:, (prompt_ids.shape[1] - 1):]

        # Sum log-probs to get total log-probability of the target answer
        total_log_prob = target_token_log_probs.sum().item()

        # After computing target_token_log_probs
        flat_logprobs = target_token_log_probs[0].tolist()
        print("Per-token logprobs:", flat_logprobs)
        print("Per-token probs:", [math.exp(lp) for lp in flat_logprobs])

        return total_log_prob


# Example usage for testing
if __name__ == "__main__":
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    from dotenv import load_dotenv
    load_dotenv('/home/hrahmani/CausalContextAttributer/config.env')
    hf_auth_token  = os.environ.get("HF_API_KEY")
    class_labels = ["positive", "negative", "neutral"]
    llama_handler = LlamaHandler(class_labels, hf_auth_token, model_name)
    llm_handler = LLM_Handler(class_labels, hf_auth_token, "meta-llama/Llama-3.2-1B")
    test_text = "Oh, fantastic, another meeting that could've been an email."
    test_label = "negative"

    print("\n--- LlamaHandler.classify_with_logprob ---")
    llama_result = llama_handler.classify_with_logprob(test_text)
    print(f"Predicted label: {llama_result['chosen_label']}")
    print(f"Unnormalized logprobs (logits): {llama_result['unnormalized_logprobs']}")
    print(f"Normalized logprobs: {llama_result['normalized_logprobs']}")

    
    generated_text, log_probs, gen_ids = llama_handler.generate_one_sentence_with_logprob(question="what is the capital of France?", context="The capital of france is paris.")
    print(f'the generated text is {generated_text} with log probability {log_probs}')
    print(f'the gen ids are {gen_ids}')
    print(llama_handler.tokenizer.encode(generated_text, add_special_tokens=False))

    # Remove EOS token if present in gen_ids
    eos_token_id = llama_handler.tokenizer.eos_token_id
    gen_ids_no_eos = gen_ids
    if gen_ids[-1] == eos_token_id:
        gen_ids_no_eos = gen_ids[:-1]

    # # Get encoding of generated text
    # encoded = llama_handler.tokenizer.encode(generated_text, add_special_tokens=False)

    # print("gen_ids (no EOS):", gen_ids_no_eos)
    # print("encoded:", encoded)
    # print("Match:", list(gen_ids_no_eos) == encoded)

    # Test compute_target_log_prob: should be near 1 for generated answer
    log_prob_generated = llama_handler.compute_target_log_prob(
        question="what is the capital of France?",
        target_answer=generated_text,
        context="The capital of France is Paris."
    )
    prob_generated = math.exp(log_prob_generated)
    print(f"Log probability of generated answer: {log_prob_generated}")
    print(f"Probability of generated answer: {prob_generated}")

    # # Test compute_target_log_prob_generate: should be near 1 for generated answer
    # log_prob_generated_step = llama_handler.compute_target_log_prob_generate(
    #     question="what is the capital of France?",
    #     target_answer=generated_text,
    #     context="The capital of France is Paris."
    # )
    # prob_generated_step = math.exp(log_prob_generated_step)
    # print(f"[Step-by-step] Log probability of generated answer: {log_prob_generated_step}")
    # print(f"[Step-by-step] Probability of generated answer: {prob_generated_step}")

    # print("\n--- LLM_Handler.get_classification_metrics ---")
    # llm_result = llm_handler.get_classification_metrics(test_text, test_label)
    # for k, v in llm_result.items():
    #     print(f"{k}: {v}")
