#%%
from transformers import LlamaTokenizer
#%%
tokenizer = LlamaTokenizer.from_pretrained('llama_tokenizer', use_fast = False) # initialize tokenizer
tokenizer.pad_token = tokenizer.eos_token

print(tokenizer.vocab_size)