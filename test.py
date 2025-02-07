import numpy as np
import re
import os
import openai
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sklearn.linear_model import LassoCV
import Solver

# Pre-compiled regex patterns
word_pattern = re.compile(r'\b\w+\b')
token_pattern = re.compile(r'\b\w+\b|\s+|[^\w\s]')

def remove_words_with_probability(prompt, probability=0.8):
    tokens = token_pattern.findall(prompt)
    modified_tokens = [
        '__' if word_pattern.fullmatch(token) and np.random.uniform(0, 1) > probability else token
        for token in tokens
    ]
    return ''.join(modified_tokens)

class Prompt_Generator():
    def __init__(self, prompt, LLM_Handler, num_datasets=5, pval=np.array([0.2, 0.4, 0.6, 0.8])):
        """
        Parameters:
            prompt: Original prompt (string)
            LLM_Handler: An instance to handle LLM completions.
            num_datasets: Number of dataset samples (M)
            pval: Array of probabilities used for sampling.
        """
        self.original_prompt = prompt
        self.pval = pval
        self.num_datasets = num_datasets
        self.sample_prompts = []
        self.LLM_Handler = LLM_Handler
        self.ps = []  # to store sampled p values

    def sample_p(self, size=None):
        return np.random.choice(self.pval, size=size)

    def coef_scaling(self):
        return (1 / (self.pval * (1 - self.pval))).mean()

    def build_vocabulary(self, originalprompt):
        """
        Builds a vocabulary (word to index) from the original prompt.
        Returns: word_to_index, unique_words, N (number of unique words)
        """
        words = re.findall(r'\b\w+\b', originalprompt)
        unique_words = []
        word_to_index = {}
        for word in words:
            if word not in word_to_index:
                word_to_index[word] = len(unique_words)
                unique_words.append(word)
        N = len(unique_words)
        return word_to_index, unique_words, N

    def create_X(self, method='uncompleted'):
        self.ps = []
        word_to_index, unique_words, N = self.build_vocabulary(self.original_prompt)
        X = np.empty((self.num_datasets, N))
        nu = self.coef_scaling()  # variance scaling parameter
        sqrt_nu = np.sqrt(nu)
        for m in range(self.num_datasets):
            p = self.sample_p()
            self.ps.append(p)
            modified_prompt = remove_words_with_probability(self.original_prompt, probability=p)
            # Precompute a row that defaults to the negative value for all words
            row = np.full(N, -1 / ((1 - p) * sqrt_nu))
            # Get the words that survived the removal; using set avoids redundant updates.
            modified_words = set(word_pattern.findall(modified_prompt))
            for word in modified_words:
                if word != "__" and word in word_to_index:
                    row[word_to_index[word]] = 1 / (p * sqrt_nu)
            X[m, :] = row
            self.sample_prompts.append(modified_prompt)
        return X

    def create_y(self, prompts, method='uncomplete'):
        # Two prompt templates are provided below.
        CLASSIFICATION_PROMPT_Base = (
            "You will be given a headline of a news article.\n"
            "Classify the article into one of the following categories: Technology, Politics, Sports, Art or others.\n"
            "Return only the name of the category, and nothing else.\n"
            "MAKE SURE your output is one of the four categories stated.\n"
            "Article headline: {prompt}"
        )
        CLASSIFICATION_PROMPT_Method2 = (
            "You will be given a headline of a news article with some blanks instead of words.\n"
            "Classify the article into one of the following categories: Technology, Politics, Sports, Art or others.\n"
            "Return only the name of the category, and nothing else.\n"
            "MAKE SURE your output is one of the four categories stated.\n"
            "Article headline: {prompt}"
        )
        y = np.zeros(len(prompts))
        outputs = []
        for i in range(len(prompts)):
            if method == 'baseline':
                API_RESPONSE = self.LLM_Handler.get_completion(
                    [{"role": "user", "content": CLASSIFICATION_PROMPT_Base.format(prompt=prompts[i])}],
                    model="gpt-4",
                    logprobs=True,
                    top_logprobs=1,
                )
            elif method == 'uncomplete':
                API_RESPONSE = self.LLM_Handler.get_completion(
                    [{"role": "user", "content": CLASSIFICATION_PROMPT_Method2.format(prompt=prompts[i])}],
                    model="gpt-4",
                    logprobs=True,
                    top_logprobs=1,
                )
            # Extract the log probability and token (this extraction assumes the structure you expect)
            val = API_RESPONSE.choices[0].logprobs.content[0].top_logprobs[0]
            outputs.append(val.token)
            y[i] = val.logprob
        return y, outputs

class LLM_Handler():
    def __init__(self) -> None:
        load_dotenv('config.env')
        openai.api_key = os.environ.get("OPENAI_API_KEY")

    def fill_in_blanks(self, prompt):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Please fill in the blanks in the following sentence with exactly one word for each blank then write the sentence with the filled words completely. Remember just sentence without any other thing and the number of filled words equal to the number of blanks."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0,
            n=1,
            stop=None
        )
        filled_prompt = response['choices'][0]['message']['content'].strip()
        return filled_prompt

    def get_completion(self,
                       messages: list[dict[str, str]],
                       model: str = "gpt-4",
                       max_tokens=500,
                       temperature=0,
                       stop=None,
                       seed=123,
                       tools=None,
                       logprobs=None,
                       top_logprobs=None,
                       ) -> str:
        params = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop,
            "seed": seed,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
        }
        if tools:
            params["tools"] = tools
        completion = openai.ChatCompletion.create(**params)
        return completion



# Define the original prompt (example headline)
original_prompt = "Local Mayor Launches Initiative to Enhance Urban Public Transport."

# Load the full design matrix and response vector.
X_full = np.load("X_M_max.npy")
y_full = np.load("y_M_max.npy")
print(f"Loaded X with shape {X_full.shape} and y with shape {y_full.shape}.")

# Define sample sizes to test.
sample_sizes = [10, 20, 50, 100, 200, 500, 1000]

# p-value array used in generating X (as in your previous code)
pval = np.array([0.2, 0.4, 0.6, 0.8])
# Compute the scaling factor.
coef_scaling = (1 / (pval * (1 - pval))).mean()

# Initialize LLM_Handler and Prompt_Generator to build the vocabulary.
llm_handler = LLM_Handler()
prompt_generator = Prompt_Generator(original_prompt, llm_handler)
word_to_index, unique_words, N = prompt_generator.build_vocabulary(original_prompt)

# Initialize a dictionary to hold coefficient trajectories for each word.
coeff_trajectories = {word: [] for word in unique_words}



# Compute LASSO coefficients for each sample size.
for M in sample_sizes:
    # Subset the arrays: first M rows.
    X_subset = X_full[:M, :]
    y_subset = y_full[:M]
    
    # Lasso solver: fit and extract coefficients.
    lasso_solver = Solver.LassoSolver(coef_scaling=coef_scaling)
    lasso_coef = lasso_solver.fit(X_subset, y_subset)
    
    # For each word (using its index), store the estimated coefficient.
    for word, idx in word_to_index.items():
        coeff_trajectories[word].append(lasso_coef[idx])

# Plot the trajectories for each word.
plt.figure(figsize=(10, 6))
for word, coeffs in coeff_trajectories.items():
    plt.plot(sample_sizes, coeffs, marker='o', label=word)
plt.xlabel("Sample Size (M)")
plt.ylabel("Estimated Coefficient")
plt.title("Word Coefficient Trajectories vs Sample Size (Lasso)")
plt.legend()
plt.show()

# # For each sample size, subset the full data and run both solvers.
# for M in sample_sizes:
#     # Subset the arrays: first M rows.
#     X_subset = X_full[:M, :]
#     y_subset = y_full[:M]
    
#     # Lasso solver: fit and extract coefficient for source 0.
#     lasso_solver = Solver.LassoSolver(coef_scaling=coef_scaling)
#     lasso_coef = lasso_solver.fit(X_subset, y_subset)
#     lasso_estimates[word] = lasso_coef
    


# # Plot the trajectories
# plt.figure(figsize=(10, 6))
# for word, coeffs in coeff_trajectories.items():
#     plt.plot(sample_sizes, coeffs, marker='o', label=word)
# plt.xlabel("Sample Size (M)")
# plt.ylabel("Estimated Coefficient")
# plt.title("Word Coefficient Trajectories vs Sample Size")
# plt.legend()
# plt.grid(True)
# plt.show()





# # =========================
# # Plot the Trajectories
# # =========================

# plt.figure(figsize=(10, 6))
# plt.plot(sample_sizes, lasso_estimates, marker='o', label='LassoSolver (sou)')
# plt.xlabel("Sample Size (M)")
# plt.ylabel("Estimated Effect")
# plt.title("Comparison of Lasso and Orthogonal Solver Estimates vs. Sample Size")
# plt.legend()
# plt.grid(True)
# plt.show()
