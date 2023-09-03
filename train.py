#%%
import copy

from dataclasses import dataclass
from torch.utils.data import DataLoader
from schedulers import CosineWDSchedule, WarmupCosineSchedule
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
    ext_n_embd = 384,
    n_embd = 256,
    dropout = 0.0,
    bias = True, # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better on small datasets
)

#%%
max_context_mask_ratio = 0.8#1.0 # how much of the input should be included in the context
max_target_mask_ratio = .25# how much of the input should be used for targets
target_block_num = 4


#%%
tokenizer = LlamaTokenizer.from_pretrained('llama_tokenizer', use_fast = False) # initialize tokenizer
tokenizer.pad_token = tokenizer.eos_token

context_encoder = Encoder(small_encoder_config) # initialize context encoder
target_encoder = copy.deepcopy(context_encoder) # create target encoder as a copy of context encoder
# freeze target encoder
for param in target_encoder.parameters():
    param.requires_grad = False
predictor = Predictor(predictor_config) # initialize predictor



#%%
dataset = load_from_disk('data/TinyStories').filter(lambda x: tokenizer(x['text'], return_tensors="pt")['input_ids'].shape[1] > 8, num_proc=12)


#%%
train_dataloader = DataLoader(dataset['train'], batch_size = 4, shuffle = True)


#%%
@dataclass
class OptimizerConfig:
    num_epochs = 100
    ema = (0.996, 1.0)
    ipe_scale = 1.0
    final_lr = 1.0e-06
    final_weight_decay = 0.4
    lr = 0.001
    start_lr = 0.0002
    warmup = 40
    weight_decay = 0.04

#%%
opt_config = OptimizerConfig()

# init momentum scheduler
ema = opt_config.ema#opt_config.ema
ipe_scale = opt_config.ipe_scale
iterations_per_epoch = len(dataset['train'])
num_epochs = opt_config.num_epochs
final_lr = opt_config.final_lr
final_wd = opt_config.final_weight_decay
lr = opt_config.lr
start_lr = opt_config.start_lr
warmup = opt_config.warmup
wd = opt_config.weight_decay

batch_size = 1

dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
use_bfloat16 = False



#%%

param_groups = [
        {
            'params': (p for n, p in context_encoder.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in context_encoder.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }
    ]

optimizer = torch.optim.AdamW(param_groups)
scheduler = WarmupCosineSchedule(
    optimizer,
    warmup_steps=int(warmup*iterations_per_epoch),
    start_lr=start_lr,
    ref_lr=lr,
    final_lr=final_lr,
    T_max=int(ipe_scale*num_epochs*iterations_per_epoch))
wd_scheduler = CosineWDSchedule(
    optimizer,
    ref_wd=wd,
    final_wd=final_wd,
    T_max=int(ipe_scale*num_epochs*iterations_per_epoch))
scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None

momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(iterations_per_epoch*num_epochs*ipe_scale)
                          for i in range(int(iterations_per_epoch*num_epochs*ipe_scale)+1))

#%%
for epoch in range(1, num_epochs + 1):
    for batch in train_dataloader:
        tokenized = tokenizer(batch['text'], padding = True, truncation = False, max_length = 1024, return_tensors="pt")

        batch_count = tokenized['input_ids'].shape[0]
        input_length = tokenized['input_ids'].shape[1]
        prediction_count = batch_count * target_block_num

        # compute the target embeddings
        target_embeddings = target_encoder(tokenized['input_ids']) # get target embeddings, no need to provide input indices.
        #target_embeddings = F.layer_norm(target_embeddings, (target_embeddings.size(-1),))  #TODO: may not be required normalize over feature-dim
        #TODO: normalize the target embeddings?

        # create the indices for the target blocks
        true_input_lengths = torch.sum(tokenized['attention_mask'], dim = 1) # get the true input length for each sample
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
        # get the target blocks embeddings in the shape of (batch_size, target_block_num, block_size, embedding_size)
        target_blocks_embeddings = torch.stack([target_embeddings[index,target_block_range,:] for index, target_block_range in enumerate(target_block_indices)]) # get the target embeddings
        
        # make sure the target blocks embeddings are correctly selected
        for index, sample_range in enumerate(target_block_indices):
            for jndex, _range in enumerate(sample_range):
                assert torch.all(target_embeddings[index, _range, :] == target_blocks_embeddings[index, jndex, :]) # make sure the target blocks embeddings are correctly selected
        # target embeddings are arranged in 
        # ((batch_sample_0, target_block_0, embedding_size),
        # (batch_sample_0, target_block_1, embedding_size),
        # (batch_sample_0, target_block_2, embedding_size), 
        # etc.)


        allowed_in_context = tokenized['attention_mask'].bool() # create a tensor of trues, representing the allowed inputs in the context
        for index, target_block_range in enumerate(target_block_indices.view(batch_count, -1)):
            allowed_in_context[index,target_block_range] = False # set the indices of the target blocks to false

        # make sure all context blocks have the same length
        context_lengths = torch.sum(allowed_in_context, dim = 1) # get the context lengths
        max_allowed_context_lengths = torch.clamp(context_lengths * max_context_mask_ratio, min = 1).int() # compute the max allowed context lengths
        max_context_length = torch.max(max_allowed_context_lengths) # get the max allowed context length

        context_inputs = torch.zeros((batch_count, max_context_length), dtype = torch.long) # create a tensor of zeros for the context inputs
        context_attention_mask = torch.zeros((batch_count, max_context_length), dtype = torch.bool) # create a tensor of zeros for the context attention mask))
        context_blocks_indices = torch.zeros((batch_count, max_context_length), dtype = torch.long) # create a tensor of zeros for the context blocks indices

        for index, allowed_context_length in enumerate(max_allowed_context_lengths): # for each sample
            context_block_indices = torch.arange(0, input_length)[allowed_in_context[index]] # get the indices of the context inputs
            perm = torch.randperm(context_block_indices.size(0)) # shuffle the indices
            idx = perm[:allowed_context_length] # select indices up to the smallest context length
            # context_blocks_indices.append(context_block_indices[idx]) # add the indices to the list
            context_blocks_indices[index, :allowed_context_length] = context_block_indices[idx] # set the context blocks indices
            context_inputs[index, :allowed_context_length] = tokenized['input_ids'][index, context_block_indices[idx]] # set the context inputs
            context_attention_mask[index, :allowed_context_length] = True # set the attention mask to true for the context inputs

        # get the context blocks embeddings
        context_blocks_embeddings = context_encoder(context_inputs, id_indices=context_blocks_indices, attn_mask=context_attention_mask.unsqueeze(1).unsqueeze(1))

        # create the prediction attention mask
        prediction_input_size = max_context_length + max_target_block_size 
        # add extra 0s to the context attention mask representing the target input attentions 
        pred_context_attn_mask = torch.cat((context_attention_mask, torch.zeros((batch_count, max_target_block_size), dtype = torch.bool)), dim = 1)
        # repeat the context attention mask for the prediction input size. We are doing it this way so that the target inputs only attend to the context inputs but no paddings
        pred_context_attn_mask = pred_context_attn_mask.repeat(1,prediction_input_size)
        # reshape the context attention mask to the correct attention matrix
        pred_context_attn_mask = pred_context_attn_mask.view(batch_count, prediction_input_size,  prediction_input_size)
        # create the target attention mask
        pred_target_attn_mask = torch.zeros((batch_count, prediction_input_size), dtype = torch.bool)
        # let the target blocks attend to each other
        for index, block_size in enumerate(target_block_sizes):
            pred_target_attn_mask[index, max_context_length:max_context_length+block_size] = True

        # add the context and target attention masks together. The targets attend to the context and themselves, the context blocks only attend to themselves
        prediction_attn_mask = pred_context_attn_mask + pred_target_attn_mask[:, None, :] * pred_target_attn_mask[:, :, None]
        # we interleave the prediction attention mask
        prediction_attn_mask = prediction_attn_mask.repeat_interleave(target_block_num, dim = 0)

        # predict target blocks from context blocks
        mask_token_embedding = F.normalize(torch.ones(small_encoder_config.n_embd), dim = 0) #TODO: replace dummy embedding, initialize the mask token embedding
        # mask_toke_embeddings are the same so we can just repeat them
        prediction_embeddings = mask_token_embedding.repeat(target_block_num * batch_count, max_target_block_size, 1) # for the number of target blocks, for the number of targets per block, repeat the mask token 
        context_embeddings = context_blocks_embeddings.repeat_interleave(target_block_num, dim = 0) #.repeat(target_block_num, 1, 1) # for the number of target blocks, repeat the context embeddings
        input_embeddings = torch.cat((context_embeddings, prediction_embeddings), dim = 1) # concatenate the context and prediction embeddings

        input_indices = torch.cat((context_blocks_indices.repeat_interleave(target_block_num, dim = 0), target_block_indices.view(prediction_count, -1)), dim = 1) # concatenate the context and target indices

        predicted_embeddings = predictor(input_embeddings, input_indices, attn_mask=prediction_attn_mask.unsqueeze(1)) # predict the target embeddings

        _, context_length, _ = context_embeddings.shape
        predicted_target_embeddings = predicted_embeddings[:,context_length:] # remove the context predictions

        # for each sample in the batch, mask out the target blocks which are irrelevant
        relevant_target_attn_mask = pred_target_attn_mask[:, context_length:].repeat_interleave(target_block_num, dim = 0).unsqueeze(2)
        # compute the loss of the masked predictions
        loss = F.smooth_l1_loss(predicted_target_embeddings * relevant_target_attn_mask, target_blocks_embeddings.view(predicted_target_embeddings.shape) * relevant_target_attn_mask) # compute the loss

        print(loss.item())
        # break if loss is nan
        assert not torch.isnan(loss), 'loss is nan!'


        if use_bfloat16:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        # grad_stats = grad_logger(encoder.named_parameters())
        optimizer.zero_grad()

        with torch.no_grad():
            m = next(momentum_scheduler)
            for param_q, param_k in zip(context_encoder.parameters(), target_encoder.parameters()):
                param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)


#%%
