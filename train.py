#%%
import copy

from dataclasses import dataclass
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
target_encoder = copy.deepcopy(context_encoder)

#%%
# freeze target encoder
for param in target_encoder.parameters():
    param.requires_grad = False

#%%
@dataclass
class OptimizerConfig:
    ema = (0.996, 1.0)
    ipe_scale = 1.0
    num_epochs = 100

#%%
# opt_config = OptimizerConfig(
#     ema = (0.996, 1.0),
#     ipe_scale = 1.0,
#     num_epochs = 100
# ) 

#%%
# init momentum scheduler
ema = (0.996, 1.0)#opt_config.ema
ipe_scale = 1.0 #opt_config.ipe_scale
ipe = 1
num_epochs = 100 #opt_config.num_epochs
batch_size = 1

momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1))

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
for text in dataset['train'][:5]['text']:
    print( tokenizer(text, return_tensors="pt")["input_ids"].shape )
    print(text)

#%%
tokenizer.pad_token = tokenizer.eos_token

tokenized = tokenizer(dataset['train'][1:3]['text'], padding = True, truncation = False, max_length = 1024, return_tensors="pt")

true_input_lengths = torch.sum(tokenized['attention_mask'], dim = 1) # get the true input length for each sample
min_length = torch.min(true_input_lengths) # get the length of the smallest sample
block_size = int(min_length * mask_ratio / target_block_num) # compute the block size based on the mask ratio and smallest input #number of inputs
target_block_indices = torch.cat([torch.randint(0, high  - block_size, (target_block_num,)) for high in true_input_lengths])# select target block indices
#%%
# compute the target embeddings
target_embeddings = target_encoder(tokenized['input_ids']) # get target embeddings, no need to provide input indices.

#%%
# create the ranges for the target blocks
target_block_ranges = torch.stack([torch.arange(block_start, block_start + block_size) for block_start in target_block_indices]).view(target_embeddings.shape[0], target_block_num, block_size) 

target_blocks_embeddings = torch.stack([target_embeddings[index,target_block_range,:] for index, target_block_range in enumerate(target_block_ranges)]) # get the target embeddings
target_blocks_embeddings.shape
#%%
# target_blocks_embeddings = target_embeddings[target_block_ranges.view(target_embeddings.shape[0], target_block_num, block_size), :, :] # TODO: probably doesn't work for 1 < dimensions get the target embeddings
# #%%
# target_blocks_embeddings = target_embeddings[:,target_block_ranges,:] # TODO: probably doesn't work for 1 < dimensions get the target embeddings
# target_blocks_embeddings.shape
#%%
for index, sample_range in enumerate(target_block_ranges):
    for jndex, _range in enumerate(sample_range):
        print(index, jndex, _range)
        assert torch.all(target_embeddings[index, _range, :] == target_blocks_embeddings[index, jndex, :])

#%%
target_block_ranges.shape

#%%
target_block_ranges.view(target_embeddings.shape[0], -1).shape

#%%
allowed_in_context = tokenized['attention_mask'].bool() # create a tensor of trues
for index, target_block_range in enumerate(target_block_ranges.view(target_embeddings.shape[0], -1)):
    allowed_in_context[index,target_block_range] = False # set the indices of the target blocks to false


print(allowed_in_context)
print(allowed_in_context.shape)

#%%
# make sure all context blocks have the same length
context_lengths = torch.sum(allowed_in_context, dim = 1) # get the context lengths
smallest_context_length = torch.min(context_lengths) # get the smallest context length
inputs_to_remove = torch.clamp(context_lengths - torch.min(context_lengths), min = 0) # compute the number of inputs to remove

context_blocks_indices = [] 
for index, input_to_remove in enumerate(inputs_to_remove): # for each sample
    context_block_indices = torch.arange(0, target_embeddings.shape[1])[allowed_in_context[index]] # get the indices of the context inputs
    perm = torch.randperm(context_block_indices.size(0)) # shuffle the indices
    idx = perm[:smallest_context_length] # select smallest_context_length indices
    context_blocks_indices.append(context_block_indices[idx]) # add the indices to the list

context_blocks_indices = torch.stack(context_blocks_indices)
#%%

#%%
context_blocks_embeddings = context_encoder(torch.gather(tokenized['input_ids'], 1, context_blocks_indices), id_indices=context_blocks_indices) 

#%%
context_blocks_embeddings.shape

#%%
target_embeddings.shape

#%%
# predict target blocks from context blocks
mask_token_embedding = F.normalize(torch.ones(small_encoder_config.n_embd), dim = 0) # initialize the mask token embedding
mask_token_embedding.shape

#%%
prediction_embeddings = mask_token_embedding.repeat(target_block_num * context_blocks_embeddings.shape[0], block_size, 1) # for the number of target blocks, for the number of targets per block, repeat the mask token 
context_embeddings = context_blocks_embeddings.repeat(target_block_num, 1, 1) # for the number of target blocks, repeat the context embeddings
context_embeddings.shape

#%%

#%%
context_blocks_indices.shape

#%%
prediction_embeddings.shape

#%%
input_embeddings = torch.cat((context_embeddings, prediction_embeddings), dim = 1) # concatenate the context and prediction embeddings
input_embeddings.shape
#%%
r_context_block_indices = context_blocks_indices.repeat(target_block_num, 1) # for the number of target blocks, repeat the context indices
r_context_block_indices.shape

#%%
target_block_ranges.view(context_embeddings.shape[0], -1).shape

#%%
input_indices = torch.cat((r_context_block_indices, target_block_ranges.view(context_embeddings.shape[0], -1)), dim = 1) # concatenate the context and target indices

all_predictions = predictor(input_embeddings, input_indices) # predict the target embeddings

#%%
_, context_length, _ = context_embeddings.shape

masked_predictions = all_predictions[:,context_length:] # remove the context predictions

#%%
masked_predictions.shape

#%%
target_blocks_embeddings.shape

#%%
# compute the loss of the masked predictions
loss = F.smooth_l1_loss(masked_predictions, target_blocks_embeddings.view(masked_predictions.shape)) # compute the loss

#%%
