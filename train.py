#%%
from src.models.encoder import Encoder, EncoderConfig, Predictor, PredictorConfig
from transformers import LlamaTokenizer
from datasets import load_from_disk
from tqdm import tqdm

import torch
import torch.nn.functional as F

#%%
encoder_config = EncoderConfig(
    block_size = 1024,
    vocab_size = 32000, # LLAMA tokenizer is 32000, which is a multiple of 64 for efficiency
    n_layer = 12,
    n_head = 12,
    n_embd = 768,
    dropout = 0.0,
    bias = True, # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better on small datasets
)

small_encoder_config = EncoderConfig(
    block_size = 1024,
    vocab_size = 32000, # LLAMA tokenizer is 32000, which is a multiple of 64 for efficiency
    n_layer = 8,
    n_head = 12,
    n_embd = 384,
    dropout = 0.0,
    bias = True, # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better on small datasets
)

predictor_config = PredictorConfig(
    n_layer = 4,
    n_head = 8,
    n_embd = 384,
    dropout = 0.0,
    bias = True, # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better on small datasets
)

#%%
mask_ratio = .25# 408 #int(512 * .4)
target_block_num = 4


#%%
tokenizer = LlamaTokenizer.from_pretrained('llama_tokenizer')
context_encoder = Encoder(small_encoder_config)
target_encoder = Encoder(small_encoder_config)

#%%
predictor = Predictor(small_encoder_config)

#%%
dataset = load_from_disk('data/TinyStories')

#%%
for index, sample in enumerate(tqdm(dataset['train'])):
    tokenized = tokenizer(sample['text'], padding = False, truncation = True, max_length = 1024, return_tensors="pt")
    target_embeddings = target_encoder(tokenized['input_ids'])
    h = F.layer_norm(target_embeddings, (target_embeddings.size(-1),)) # normalize over the feature dimension

    block_size = int(target_embeddings.shape[1] * mask_ratio / target_block_num)
    print(block_size)
    target_block_indices = torch.randint(0, target_embeddings.shape[1] - block_size, (target_block_num,))   
    target_block_ranges = torch.stack([target_block_indices, target_block_indices + block_size], dim = 1)

    target_blocks_embeddings = [target_embeddings[target_block_range[0]:target_block_range[1]] for target_block_range in target_block_ranges]

    masked_ids = torch.clone(tokenized['input_ids'])
    # target_block_mask = torch.zeros((target_embeddings.shape[1]), dtype = torch.bool)
    for target_index, target_block_range in enumerate(target_block_ranges):
        masked_ids[0,target_block_range[0]:target_block_range[1]] = tokenizer.unk_token_id

    print(sum(masked_ids[0] == tokenizer.unk_token_id) / masked_ids.shape[1])

    print(target_block_ranges)
    print(tokenizer.decode(tokenized['input_ids'][0]))
    print(tokenizer.decode(masked_ids[0]))
    # target_block_mask = torch.zeros(target_embeddings.shape[1], dtype = torch.bool)

    # for target_block_range in target_block_ranges:
    #     target_block_mask[target_block_range[0]:target_block_range[1]] = True

    # targets = target_embeddings[target_block_ranges]

    # sample target blocks with their size ranging from 128 to 256

     

    print(target_embeddings.shape)

    if index == 10:
        break

#%%
tokenized = tokenizer(sample['text'], padding = False, truncation = True, max_length = 1024, return_tensors="pt")
target_embeddings = target_encoder(tokenized['input_ids'])

#%%
block_size = int(target_embeddings.shape[1] * mask_ratio / target_block_num)
print(block_size)
target_block_indices = torch.randint(0, target_embeddings.shape[1] - block_size, (target_block_num,))   
target_block_ranges = torch.stack([torch.arange(block_start, block_start + block_size) for block_start in target_block_indices], dim = 0)
target_blocks_embeddings = target_embeddings[:,target_block_ranges,:] # TODO: probably doesn't work for 1 < dimensions

#%%
# get the indices fot the encode which are the inverted indices of the target blocks
allowed_in_context = torch.ones((target_embeddings.shape[1],), dtype = torch.bool) # create a 
allowed_in_context[target_block_ranges] = False
context_block_indices = torch.arange(0, target_embeddings.shape[1]).unsqueeze(0)

context_block_indices = context_block_indices[:,allowed_in_context] # TODO: probably doesn't work for 1 < dimensions
context_blocks_embeddings = context_encoder(torch.gather(tokenized['input_ids'], 1, context_block_indices), id_indices=context_block_indices) 

#%%
print(torch.gather(tokenized['input_ids'], 1, context_block_indices).shape)
print(context_block_indices.shape)

#%%
# predict target blocks from context blocks
mask_token_embedding = F.normalize(torch.ones(small_encoder_config.n_embd), dim = 0) # initialize the mask token embedding
mask_token_embedding.shape

#%%
prediction_embeddings = mask_token_embedding.repeat(target_block_num, block_size, 1) # for the number of target blocks, for the number of targets per block, repeat the mask token 
context_embeddings = context_blocks_embeddings.repeat(target_block_num, 1, 1) # for the number of target blocks, repeat the context embeddings

#%%
input_embeddings = torch.cat((context_embeddings, prediction_embeddings), dim = 1) # concatenate the context and prediction embeddings

#%%
context_block_indices = context_block_indices.repeat(target_block_num, 1) # for the number of target blocks, repeat the context indices
input_indices = torch.cat((context_block_indices, target_block_ranges), dim = 1) # concatenate the context and target indices
all_predictions = predictor(input_embeddings, input_indices) # predict the target embeddings

#%%
_, context_length, _ = context_blocks_embeddings.shape

masked_predictions = all_predictions[:,context_length:] # remove the context predictions

#%%
# compute the loss of the masked predictions
loss = F.smooth_l1_loss(masked_predictions.unsqueeze(0), target_blocks_embeddings)

#%%
masked_predictions.unsqueeze(0).shape

#%%
torch.gather(tokenized['input_ids'], 1, context_block_indices).shape

#%%
input_embeddings.shape

#%%
input_indices.shape
#%%
prediction_embeddings.shape
# target_embeddings[0].expand(target_block_num, 1, 1)


#%%
context_block_indices
# targets = [target_embeddings[target_block_range[0]:target_block_range[1]] for target_block_range in target_block_ranges]

# #%%
# block_size = int(target_embeddings.shape[1] * mask_ratio)
# target_block_indices = torch.randint(0, target_embeddings.shape[1] - block_size, (target_block_num,))   
# target_block_ranges = torch.stack([target_block_indices, target_block_indices + block_size], dim = 1)






# # model.get_num_params()

# #%%
# prompt = "Hello world!\n How are you doing today?"

# #%%
# encoded = tokenizer(prompt, return_tensors="pt")

# #%%
# # output = model(encoded['input_ids'])

# #%%
# encoded['input_ids'].shape
# #%%
# # output.shape

# #%%
# # get tokenizer vocab size
# dataset = load_from_disk('data/TinyStories')