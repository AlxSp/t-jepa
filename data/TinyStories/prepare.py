#%%
import os

import torch
# get file path
file_path = os.path.abspath(__file__)

dataset_dir = os.path.dirname(file_path)
# go to project directory
os.chdir(os.path.dirname(os.path.dirname(dataset_dir)))

from datasets import load_dataset#load_from_disk
from tqdm import tqdm
from transformers import LlamaTokenizer
#%%
# Load the dataset
#%%
def tokenize_and_cast(examples):
    # Tokenize the examples
    tokenized_examples = tokenizer(examples['text'], return_tensors="pt", add_special_tokens=False, return_attention_mask=False)
    # Ensure all outputs are of the same dtype, e.g., convert to float32 or as needed
    tokenized_examples['input_ids'] = tokenized_examples['input_ids'].to(dtype=torch.float32)
    # Add any other necessary conversions here
    return tokenized_examples

# if __name__ == '__main__':
#%%
num_proc = 4

tokenizer = LlamaTokenizer.from_pretrained('llama_tokenizer')

dataset = load_dataset("roneneldan/TinyStories", cache_dir=dataset_dir, num_proc=num_proc)
#%%
tokenized = dataset.map(tokenize_and_cast, num_proc=num_proc)
#%%
tokenized = tokenized.filter(lambda x: torch.tensor(x['input_ids']).shape[1] >= 64, num_proc=num_proc)

#%%
# remove the input_ids from the tokenized dataset
tokenized = tokenized.remove_columns('input_ids')
#%%
# save the tokenized dataset
tokenized.save_to_disk(dataset_dir)

    






# %%
