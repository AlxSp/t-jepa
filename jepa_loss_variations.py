#%%
import copy
import random
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import LlamaTokenizer
import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl
import matplotlib.patheffects as pe

from src.models.encoder import Encoder, EncoderConfig, Predictor, PredictorConfig

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
    ext_n_embd = 384,
    n_embd = 256,
    dropout = 0.0,
    bias = True, # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better on small datasets
)

#%%
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

#%%
random_seed = 42

#%%
max_context_mask_ratio = 0.8#1.0
max_target_mask_ratio = .25# 408 #int(512 * .4)
target_block_num = 4

device = 'cpu'

#%%
tokenizer = LlamaTokenizer.from_pretrained('llama_tokenizer') # initialize tokenizer
context_encoder = Encoder(small_encoder_config) # initialize context encoder
target_encoder = copy.deepcopy(context_encoder) # create target encoder as a copy of context encoder

predictor = Predictor(predictor_config) # initialize predictor

#%%
# freeze target encoder
for param in target_encoder.parameters():
    param.requires_grad = False

#%%
dataset = load_from_disk('data/TinyStories')

#%%
tokenizer.pad_token = tokenizer.eos_token

batch = dataset['train'][1:4]

tokenized = tokenizer(batch['text'], padding = True, truncation = False, max_length = 1024, return_tensors="pt", add_special_tokens = False)

input_ids = tokenized['input_ids']
input_attn_mask = tokenized['attention_mask']


#%%
set_seed(random_seed)


#%%
batch_count = input_ids.shape[0]
input_length = input_ids.shape[1]
prediction_count = batch_count * target_block_num

target_embeddings = target_encoder(input_ids.to(device)) # get target embeddings, no need to provide input indices.
#target_embeddings = F.layer_norm(target_embeddings, (target_embeddings.size(-1),))  #TODO: may not be required normalize over feature-dim
#TODO: normalize the target embeddings?

# create the indices for the target blocks
true_input_lengths = torch.sum(input_attn_mask, dim = 1) # get the true input length for each sample
target_block_sizes = torch.clamp((true_input_lengths * max_target_mask_ratio / target_block_num), 1).int() # compute the target block sizes for each sample
max_target_block_size = max(target_block_sizes) # get the maximum target block size

# TODO: add feature to restrict min distance spacing between target blocks
target_block_start_indices = torch.cat([torch.randperm(high  - block_size).repeat(2)[:target_block_num] for high, block_size in zip(true_input_lengths, target_block_sizes)]) # for each sample get the starting index of the n target blocks
target_block_indices = torch.ones((batch_count, target_block_num, max_target_block_size), dtype = torch.long) # create a tensor to hold the indices of the target blocks
target_block_indices = target_block_indices * target_block_start_indices.view((batch_count, target_block_num, 1)) # multiply the starting index of each target block by the full index tensor

# for each sample add the indices of the target blocks to the tensor by applying a range to each row of the index tensors
for batch_idx in range(batch_count):
    # if the target block size is smaller than the max target block size, then the indices past the target block size will not be updated (they are handled by the attention mask)
    target_block_indices[batch_idx, :, :target_block_sizes[batch_idx]] += torch.arange(target_block_sizes[batch_idx]).view(1, -1) # add the indices of the target blocks to the tensor

print("target_block_indices")
print(target_block_indices)

print("target_block_input_ids")
print(torch.stack([input_ids[index,target_block_range] for index, target_block_range in enumerate(target_block_indices)]))

print("target block tokens")
print([tokenizer.batch_decode(target_blocks_ids) for target_blocks_ids in torch.stack([input_ids[index,target_block_range] for index, target_block_range in enumerate(target_block_indices)])])

#%%

#%%
# get the target blocks embeddings in the shape of (batch_size, target_block_num, block_size, embedding_size)
target_blocks_embeddings = torch.stack([target_embeddings[index,target_block_range,:] for index, target_block_range in enumerate(target_block_indices)]) # get the target embeddings

# make sure the target blocks embeddings are correctly selected
for batch_index, sample_range in enumerate(target_block_indices):
    for jndex, _range in enumerate(sample_range):
        assert torch.all(target_embeddings[batch_index, _range, :] == target_blocks_embeddings[batch_index, jndex, :]) # make sure the target blocks embeddings are correctly selected
# target embeddings are arranged in 
# ((batch_sample_0, target_block_0, embedding_size),
# (batch_sample_0, target_block_1, embedding_size),
# (batch_sample_0, target_block_2, embedding_size), 
# etc.)


allowed_in_context = input_attn_mask.bool() # create a tensor of trues, representing the allowed inputs in the context
for batch_index, target_block_range in enumerate(target_block_indices.view(batch_count, -1)):
    allowed_in_context[batch_index,target_block_range] = False # set the indices of the target blocks to false

# make sure all context blocks have the same length
context_lengths = torch.sum(allowed_in_context, dim = 1) # get the context lengths
max_allowed_context_lengths = torch.clamp(context_lengths * max_context_mask_ratio, min = 1).int() # compute the max allowed context lengths
max_context_length = torch.max(max_allowed_context_lengths) # get the max allowed context length

context_inputs = torch.zeros((batch_count, max_context_length), dtype = torch.long) # create a tensor of zeros for the context inputs
context_attention_mask = torch.zeros((batch_count, max_context_length), dtype = torch.bool) # create a tensor of zeros for the context attention mask))
context_blocks_indices = torch.zeros((batch_count, max_context_length), dtype = torch.long) # create a tensor of zeros for the context blocks indices

set_seed(random_seed)
for batch_index, allowed_context_length in enumerate(max_allowed_context_lengths): # for each sample
    context_block_indices = torch.arange(0, input_length)[allowed_in_context[batch_index]] # get the indices of the context inputs
    perm = torch.randperm(context_block_indices.size(0)) # shuffle the indices
    idx = perm[:allowed_context_length] # select indices up to the smallest context length
    # context_blocks_indices.append(context_block_indices[idx]) # add the indices to the list
    context_blocks_indices[batch_index, :allowed_context_length] = context_block_indices[idx] # set the context blocks indices
    context_inputs[batch_index, :allowed_context_length] = input_ids[batch_index, context_block_indices[idx]] # set the context inputs
    context_attention_mask[batch_index, :allowed_context_length] = True # set the attention mask to true for the context inputs

#%%
print("context inputs")
context_input_ids = torch.zeros_like(input_ids)
for batch_index, context_block_range in enumerate(context_blocks_indices):
    msk = input_attn_mask[batch_index]
    context_input_ids[batch_index, context_block_range[:torch.sum(msk)]] = context_inputs[batch_index][:torch.sum(msk)]
print(context_inputs)
print([tokenizer.batch_decode(ids) for ids in context_input_ids])


#%%
# get the context blocks embeddings
context_blocks_embeddings = context_encoder(context_inputs.to(device), id_indices=context_blocks_indices.to(device), attn_mask=context_attention_mask.unsqueeze(1).unsqueeze(1).to(device))

# create the prediction attention mask
prediction_input_size = max_context_length + max_target_block_size 
# add extra 0s to the context attention mask representing the target input attentions 
pred_context_attn_mask = torch.cat((context_attention_mask, torch.zeros((batch_count, max_target_block_size), dtype = torch.bool)), dim = 1)
#%%
# create the target attention mask
pred_target_attn_mask = torch.zeros((batch_count, prediction_input_size), dtype = torch.bool)
# let the tokens in each target block attend to each other
for batch_index, block_size in enumerate(target_block_sizes):
    pred_target_attn_mask[batch_index, max_context_length:max_context_length+block_size] = True

#%%
# add the context and target attention masks together. The targets attend to the context and themselves, the context blocks only attend to themselves
prediction_attn_mask = pred_context_attn_mask[:, None, :] * pred_context_attn_mask[:, :, None] + pred_context_attn_mask[:, None, :] * pred_target_attn_mask[:, :, None] + pred_target_attn_mask[:, None, :] * pred_target_attn_mask[:, :, None]
# we interleave the prediction attention mask
prediction_attn_mask = prediction_attn_mask.repeat_interleave(target_block_num, dim = 0)

# predict target blocks from context blocks
mask_token_embedding = predictor.get_mask_token_embedding()
# mask_toke_embeddings are the same so we can just repeat them
prediction_embeddings = mask_token_embedding.repeat(target_block_num * batch_count, max_target_block_size, 1) # for the number of target blocks, for the number of targets per block, repeat the mask token 
context_embeddings = context_blocks_embeddings.repeat_interleave(target_block_num, dim = 0) #.repeat(target_block_num, 1, 1) # for the number of target blocks, repeat the context embeddings
input_embeddings = torch.cat((context_embeddings, prediction_embeddings.to(device)), dim = 1) # concatenate the context and prediction embeddings

input_indices = torch.cat((context_blocks_indices.repeat_interleave(target_block_num, dim = 0), target_block_indices.view(prediction_count, -1)), dim = 1) # concatenate the context and target indices

print("prediction_input_indices")
print(input_indices)

#%%
z_prediction_attn_mask = prediction_attn_mask.clone()

for i in range(z_prediction_attn_mask.shape[0]):
    ctx_len = torch.sum(z_prediction_attn_mask[i], dim = 1)[0]
    z_prediction_attn_mask[i, :, :1] = True
    # print(torch.sum(z_prediction_attn_mask[i], dim = 1))
plt.imshow(z_prediction_attn_mask[0].cpu().numpy())

z_predicted_embeddings = predictor(input_embeddings, input_indices.to(device), attn_mask=z_prediction_attn_mask.unsqueeze(1).to(device)) # predict the target embeddings

#%%
y_prediction_attn_mask = prediction_attn_mask.clone()

for i in range(y_prediction_attn_mask.shape[0]):
    ctx_len = torch.sum(y_prediction_attn_mask[i], dim = 1)[0]
    y_prediction_attn_mask[i, :, :ctx_len] = True
    # print(torch.sum(y_prediction_attn_mask[i], dim = 1))
plt.imshow(y_prediction_attn_mask[0].cpu().numpy())

y_predicted_embeddings = predictor(input_embeddings, input_indices.to(device), attn_mask=y_prediction_attn_mask.unsqueeze(1).to(device)) # predict the target embeddings

#%%
# use attention mask with diagonal attention
x_prediction_attn_mask = prediction_attn_mask.clone()
diagonal_indices = torch.arange(x_prediction_attn_mask.shape[1])

x_prediction_attn_mask[:, diagonal_indices, diagonal_indices] = True

plt.imshow(x_prediction_attn_mask[0].cpu().numpy())

x_predicted_embeddings = predictor(input_embeddings, input_indices.to(device), attn_mask=x_prediction_attn_mask.unsqueeze(1).to(device)) # predict the target embeddings

#%%
for index, attended_ids in enumerate(torch.diagonal(prediction_attn_mask, dim1=1, dim2=2)):
    assert torch.allclose(z_predicted_embeddings[index][attended_ids], y_predicted_embeddings[index][attended_ids]) and torch.allclose(z_predicted_embeddings[index][attended_ids], x_predicted_embeddings[index][attended_ids])

# #%%
# z_predicted_embeddings[torch.diagonal(prediction_attn_mask, dim1=1, dim2=2)].shape

# #%%
# # for index, indices in enumerate(torch.diagonal(prediction_attn_mask, dim1=1, dim2=2)):

# #     print(index, indices)
# #     print(z_predicted_embeddings[index][indices].shape)
# #     print(y_predicted_embeddings[index][indices])
# assert torch.allclose(z_predicted_embeddings[torch.diagonal(prediction_attn_mask, dim1=1, dim2=2)], y_predicted_embeddings[torch.diagonal(prediction_attn_mask, dim1=1, dim2=2)], atol=1e-5)

# #%%
#TODO: check why having ids which are not attended to at all, causes nan. this is necessary
for i in range(prediction_attn_mask.shape[0]):
    ctx_len = torch.sum(prediction_attn_mask[i], dim = 1)[0]
    prediction_attn_mask[i, :, :ctx_len] = True
#%%
predicted_embeddings = predictor(input_embeddings, input_indices.to(device), attn_mask=prediction_attn_mask.unsqueeze(1).to(device)) # predict the target embeddings

_, context_length, _ = context_embeddings.shape
a_predicted_target_embeddings = predicted_embeddings[:,context_length:] # remove the context predictions

# for each sample in the batch, mask out the target blocks which are irrelevant
relevant_target_attn_mask = pred_target_attn_mask[:, context_length:].repeat_interleave(target_block_num, dim = 0).unsqueeze(2)

relevant_target_attn_mask = relevant_target_attn_mask.to(device)

# compute the loss of the masked predictions
a_loss = F.smooth_l1_loss(a_predicted_target_embeddings * relevant_target_attn_mask, target_blocks_embeddings.view(a_predicted_target_embeddings.shape) * relevant_target_attn_mask) # compute the loss


#%%

prediction_input_ids = torch.zeros_like(input_ids)
prediction_input_ids.shape

#%%

# add the input tokens to the plot
for i in range(prediction_attn_mask.shape[0]):
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(prediction_attn_mask[i].cpu().numpy())
    ids = input_ids[i // target_block_num][input_indices[i]]

    print("len ids", len(ids))


    print("input_indices", input_indices[i])
    print("ids", ids)
    print(tokenizer.decode(ids))

    sorted_ids = []
    context_arg_indices = torch.argsort(input_indices[i][:max_allowed_context_lengths[i // target_block_num]])
    sorted_context_ids = [tokenizer.decode(ids[index]) for index in context_arg_indices]
    print(len(sorted_context_ids))
    sorted_ids.extend(sorted_context_ids)
    padding = ["#"] * (max_context_length - len(sorted_context_ids))
    print(len(padding))
    sorted_ids.extend(padding)
    target_arg_indices = torch.argsort(input_indices[i][max_context_length:max_context_length + target_block_sizes[i // target_block_num]])
    sorted_target_ids = [tokenizer.decode(ids[index + len(sorted_ids)]) for index in target_arg_indices]
    print(len(sorted_target_ids))
    sorted_ids.extend(sorted_target_ids)
    padding = ["#"] * (len(ids) - len(sorted_ids) )
    print(len(padding))
    sorted_ids.extend(padding)

    print("sorted_ids", len(sorted_ids))





    print(context_arg_indices)

    print(tokenizer.decode([ids[index] for index in context_arg_indices]))
    print(max_allowed_context_lengths[i // target_block_num])
    print(context_block_indices)

    decoded_ids = sorted_ids#[tokenizer.decode(id_) for id_ in ids]
    # for j in range(prediction_attn_mask.shape[1]):
    #     ax.text(j, j, decoded_ids[j], ha="center", va="center", color="black", path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    # ax.spines[:].set_visible(False)

    # Set the tick positions to the center of each tile
    tick_positions = torch.arange(prediction_attn_mask.shape[1])

# Set the tick positions and labels for x-axis at the top
    ax.set_xticks(tick_positions)
    ax.set_xticks(torch.arange(prediction_attn_mask.shape[1] + 1) - 0.5, minor = True)
    ax.set_xticklabels(decoded_ids, rotation=90, )
    ax.xaxis.set_ticks_position('top')


    ax.set_yticks(tick_positions)
    ax.set_yticklabels(decoded_ids)
    ax.set_yticks(torch.arange(prediction_attn_mask.shape[1] + 1) - 0.5, minor = True)

    
    
    ax.grid(which = "minor", color = "w", linestyle = '-', linewidth = .5)
    ax.tick_params(which =  "minor", bottom = False, left = False)
    fig.tight_layout()
    # break







# TARGET PACKING

#%%
target_block_indices

#%%
target_block_indices = target_block_indices.view(batch_count, -1)
target_block_indices

#%%
# get the target blocks embeddings in the shape of (batch_size, target_block_num, block_size, embedding_size)
target_blocks_embeddings = torch.stack([target_embeddings[index,target_block_range,:] for index, target_block_range in enumerate(target_block_indices)]) # get the target embeddings

# make sure the target blocks embeddings are correctly selected
for batch_index, sample_range in enumerate(target_block_indices):
    for jndex, _range in enumerate(sample_range):
        assert torch.all(target_embeddings[batch_index, _range, :] == target_blocks_embeddings[batch_index, jndex, :]) # make sure the target blocks embeddings are correctly selected
#%%
allowed_in_context = input_attn_mask.bool() # create a tensor of trues, representing the allowed inputs in the context
for batch_index, target_block_range in enumerate(target_block_indices):
    allowed_in_context[batch_index,target_block_range] = False # set the indices of the target blocks to false
allowed_in_context
#%%

# make sure all context blocks have the same length
context_lengths = torch.sum(allowed_in_context, dim = 1) # get the context lengths
max_allowed_context_lengths = torch.clamp(context_lengths * max_context_mask_ratio, min = 1).int() # compute the max allowed context lengths
max_context_length = torch.max(max_allowed_context_lengths) # get the max allowed context length

context_inputs = torch.zeros((batch_count, max_context_length), dtype = torch.long) # create a tensor of zeros for the context inputs
context_attention_mask = torch.zeros((batch_count, max_context_length), dtype = torch.bool) # create a tensor of zeros for the context attention mask))
context_blocks_indices = torch.zeros((batch_count, max_context_length), dtype = torch.long) # create a tensor of zeros for the context blocks indices

set_seed(random_seed)
for batch_index, allowed_context_length in enumerate(max_allowed_context_lengths): # for each sample
    context_block_indices = torch.arange(0, input_length)[allowed_in_context[batch_index]] # get the indices of the context inputs
    perm = torch.randperm(context_block_indices.size(0)) # shuffle the indices
    idx = perm[:allowed_context_length] # select indices up to the smallest context length
    # context_blocks_indices.append(context_block_indices[idx]) # add the indices to the list
    context_blocks_indices[batch_index, :allowed_context_length] = context_block_indices[idx] # set the context blocks indices
    context_inputs[batch_index, :allowed_context_length] = input_ids[batch_index, context_block_indices[idx]] # set the context inputs
    context_attention_mask[batch_index, :allowed_context_length] = True # set the attention mask to true for the context inputs


#%%
print("context inputs")
context_input_ids = torch.zeros_like(input_ids)
for batch_index, context_block_range in enumerate(context_blocks_indices):
    msk = input_attn_mask[batch_index]
    context_input_ids[batch_index, context_block_range[:torch.sum(msk)]] = context_inputs[batch_index][:torch.sum(msk)]
print(context_inputs)
print([tokenizer.batch_decode(ids) for ids in context_input_ids])

#%%
# get the context blocks embeddings
context_blocks_embeddings = context_encoder(context_inputs.to(device), id_indices=context_blocks_indices.to(device), attn_mask=context_attention_mask.unsqueeze(1).unsqueeze(1).to(device))
#%%
# create the prediction attention mask
prediction_input_size = max_context_length + max_target_block_size * target_block_num

#%%
pred_context_attn_mask = torch.cat((context_attention_mask, torch.zeros((batch_count, max_target_block_size * target_block_num), dtype = torch.bool)), dim = 1)

#%%
prediction_attn_mask = pred_context_attn_mask[:, None, :] * pred_context_attn_mask[:, :, None]
# let the tokens in each target block attend to each other but not the other target blocks
for batch_index, block_size in enumerate(target_block_sizes):
    for target_block_index in range(target_block_num):
        pred_target_attn_mask = torch.zeros((prediction_input_size), dtype = torch.bool)
        target_attn_start_index = max_context_length + target_block_index * max_target_block_size
        pred_target_attn_mask[target_attn_start_index:target_attn_start_index+block_size] = True

        prediction_attn_mask[batch_index] += pred_context_attn_mask[batch_index, None, :] * pred_target_attn_mask[:, None] +  pred_target_attn_mask[None, :] * pred_target_attn_mask[:, None]
#%%
for i in range(prediction_attn_mask.shape[0]):
    ctx_len = torch.sum(prediction_attn_mask[i], dim = 1)[0]
    prediction_attn_mask[i, :, :ctx_len] = True

#%%
# predict target blocks from context blocks
mask_token_embedding = predictor.get_mask_token_embedding()
# mask_toke_embeddings are the same so we can just repeat them
prediction_embeddings = mask_token_embedding.repeat(batch_count, max_target_block_size * target_block_num, 1) # for the number of target blocks, for the number of targets per block, repeat the mask token 

#%%
input_embeddings = torch.cat((context_blocks_embeddings, prediction_embeddings.to(device)), dim = 1) # concatenate the context and prediction embeddings

#%%
input_indices = torch.cat((context_blocks_indices, target_block_indices), dim = 1) # concatenate the context and target indices

#%%
predicted_embeddings = predictor(input_embeddings, input_indices.to(device), attn_mask=prediction_attn_mask.unsqueeze(1).to(device)) # predict the target embeddings

_, context_length, _ = context_embeddings.shape

#%%
relevant_target_attn_mask = torch.diagonal(prediction_attn_mask, dim1=1, dim2=2)
relevant_target_attn_mask = relevant_target_attn_mask[:, context_length:].unsqueeze(2).to(device)

b_predicted_target_embeddings = predicted_embeddings[:,context_length:] # remove the context predictions
print(b_predicted_target_embeddings.shape)
# for each sample in the batch, mask out the target blocks which are irrelevant
# relevant_target_attn_mask = pred_target_attn_mask[:, context_length:].repeat_interleave(target_block_num, dim = 0).unsqueeze(2)

# relevant_target_attn_mask = relevant_target_attn_mask[:,context_length:].unsqueeze(2).to(device)

print(relevant_target_attn_mask.shape)

#%%
relevant_target_attn_mask.shape

#%%

# compute the loss of the masked predictions
b_loss = F.smooth_l1_loss(b_predicted_target_embeddings * relevant_target_attn_mask, target_blocks_embeddings.view(b_predicted_target_embeddings.shape) * relevant_target_attn_mask) # compute the loss
#%%
# add the input tokens to the plot
for i in range(prediction_attn_mask.shape[0]):
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(prediction_attn_mask[i].cpu().numpy())
    ids = input_ids[i][input_indices[i]]

    print("len ids", len(ids))


    print("input_indices", input_indices[i])
    print("ids", ids)
    print(tokenizer.decode(ids))

    sorted_ids = []
    sorted_indices = []
    context_arg_indices = torch.argsort(input_indices[i][:max_allowed_context_lengths[i]])
    sorted_context_ids = [tokenizer.decode(ids[index]) for index in context_arg_indices]
    print(len(sorted_context_ids))
    sorted_ids.extend(sorted_context_ids)
    sorted_indices.extend([str(index.item()) for index in input_indices[i][:max_allowed_context_lengths[i]][context_arg_indices]])

    padding = ["#"] * (max_context_length - len(sorted_context_ids))
    print(len(padding))
    sorted_ids.extend(padding)
    sorted_indices.extend([""] * len(padding))

    for j in range(target_block_num):
        target_start_index = max_context_length + j * max_target_block_size
        target_arg_indices = torch.argsort(input_indices[i][target_start_index:target_start_index + target_block_sizes[i]])
        sorted_target_ids = [tokenizer.decode(ids[index + len(sorted_ids)]) for index in target_arg_indices]
        print(target_arg_indices)
        sorted_ids.extend(sorted_target_ids)
        padding = ["#"] * (max_target_block_size - target_block_sizes[i])
        print(padding)
        sorted_ids.extend(padding)
        sorted_indices.extend([str(index.item()) for index in input_indices[i][target_start_index:target_start_index + target_block_sizes[i]][target_arg_indices]])
        sorted_indices.extend([""] * len(padding))

    print("sorted_ids", len(sorted_ids))

    print(context_arg_indices)

    print(tokenizer.decode([ids[index] for index in context_arg_indices]))
    print(max_allowed_context_lengths[i])
    print(context_block_indices)

    decoded_ids = sorted_ids#[tokenizer.decode(id_) for id_ in ids]
    for j in range(prediction_attn_mask.shape[1]):
        ax.text(j, j, sorted_indices[j], ha="center", va="center", color="black", path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    ax.spines[:].set_visible(False)

    # Set the tick positions to the center of each tile
    tick_positions = torch.arange(prediction_attn_mask.shape[1])

# Set the tick positions and labels for x-axis at the top
    ax.set_xticks(tick_positions)
    ax.set_xticks(torch.arange(prediction_attn_mask.shape[1] + 1) - 0.5, minor = True)
    ax.set_xticklabels(decoded_ids, rotation=90, )
    ax.xaxis.set_ticks_position('top')


    ax.set_yticks(tick_positions)
    ax.set_yticklabels(decoded_ids)
    ax.set_yticks(torch.arange(prediction_attn_mask.shape[1] + 1) - 0.5, minor = True)

    
    
    ax.grid(which = "minor", color = "w", linestyle = '-', linewidth = .5)
    ax.tick_params(which =  "minor", bottom = False, left = False)
    fig.tight_layout()


#%%
a_predicted_target_embeddings.shape

#%%
b_predicted_target_embeddings.shape

#%%
a_predicted_target_embeddings[:4, :].reshape(-1, 384)

#%%
b_predicted_target_embeddings[0].shape

#%%
torch.diagonal(prediction_attn_mask, dim1=1, dim2=2)[0]

#%%
# for a_emb, b_emb, att_msk in zip(a_predicted_target_embeddings[:4, :].reshape(-1, 384), b_predicted_target_embeddings[0], torch.diagonal(prediction_attn_mask, dim1=1, dim2=2)[0]):
#     print(att_msk)
#     is_close = torch.allclose(a_emb, b_emb, atol=1e-7)
#     print(is_close)
#     if not is_close and att_msk.item():
#         print(a_emb)
#         print(b_emb)
#     print()


# assert torch.allclose(a_predicted_target_embeddings[:4, :].reshape(-1, 384), b_predicted_target_embeddings[0], atol=1e-6)

#%%
reshaped_a_predicted_target_embeddings = a_predicted_target_embeddings.reshape(-1, 12, 384)

for index, attended_ids in enumerate(torch.diagonal(prediction_attn_mask, dim1=1, dim2=2)):
    print(index)
    attended_ids = attended_ids[context_length:]

    for id_attn, a_emb, b_emb in zip(attended_ids, reshaped_a_predicted_target_embeddings[index][attended_ids], b_predicted_target_embeddings[index][attended_ids]):
        id_attn = id_attn.item()
        print(id_attn)
        is_close = torch.allclose(a_emb, b_emb, atol=1e-6)
        print(torch.allclose(a_emb, b_emb))
        if not is_close and id_attn:
            print(a_emb)
            print(b_emb)

        print()
    # is_close = torch.allclose(reshaped_a_predicted_target_embeddings[index][attended_ids], b_predicted_target_embeddings[index][attended_ids], atol=1e-6)

    # print()




