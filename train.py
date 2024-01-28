#%%
import copy
# import the library
from dataclasses import dataclass
import math
import os
import json
from contextlib import nullcontext
import random
import shutil
import sys
import numpy as np
from torch.utils.data import DataLoader
from schedulers import CosineWDSchedule, ExponentialMovingAverageSchedule, WarmupCosineSchedule
from src.models.encoder import Encoder, EncoderConfig, Predictor, PredictorConfig
from transformers import LlamaTokenizer
from datasets import load_from_disk
from tqdm import tqdm

import torch
import torch.nn.functional as F

#%%
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def dataclass_to_json(dataclass, path):
    with open(path, 'w') as f:
        json.dump(dataclass.__dict__, f, indent=2)

def json_to_dataclass(dataclass, path):
    with open(path, 'r') as f:
        data = json.load(f)
    return dataclass(**data)

#%%
# encoder_config = EncoderConfig(
#     block_size = 1024,
#     vocab_size = 32000, # LLAMA tokenizer is 32000, which is a multiple of 64 for efficiency
#     n_layer = 12,
#     n_head = 12,
#     n_embd = 768,
#     dropout = 0.0,
#     bias = True, # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better on small datasets
# )

encoder_config = EncoderConfig(
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
    trainable_mask_emb=False
)

@dataclass
class TrainRunConfig:
    batch_size: int = 40
    accumulation_steps: int = 4
    eval_interval: int = 40
    num_eval_batches: int = 20
    max_iter_num: int | None = 100
    random_seed: int = 42

train_run_config = TrainRunConfig(
    batch_size = 64,
    accumulation_steps=32,
    eval_interval=1024,
    num_eval_batches = 200,
    max_iter_num=100,
    random_seed=42
)


@dataclass
class OptimizerConfig:
    num_epochs: int = 100
    ema: tuple[float, float] = (0.996, 1.0)
    bipe_scale: float = 1.0
    final_lr: float = 1.0e-06
    final_weight_decay: float = 0.4
    lr: float = 0.001
    start_lr: float = 0.0002
    warmup: int = 40
    weight_decay: float = 0.04

#%%
opt_config = OptimizerConfig(
    num_epochs = 100,
    ema = (0.996, 1.0),
    bipe_scale = 1.0,
    final_lr = 1.0e-06,
    final_weight_decay = 0.4,
    lr = 0.001,
    start_lr = 0.0002,
    warmup = 40,
    weight_decay = 0.04
)

#%%
wandb_log = True
wandb_project = "t-jepa"
wandb_run_name = "t-jepa"

init_from = "resume"
init_from = "scratch"
init_from = init_from if not sys.argv[1:] else sys.argv[1]
resume_from = "train" # "train" or "best"

print(f"init from: {init_from}")

out_dir = "out"
train_out_dir = os.path.join(out_dir, "train")

max_iter_num = None if not sys.argv[2:] else int(sys.argv[2])

# best eval paths
context_encoder_path = os.path.join(out_dir, "context_encoder.pt")
target_encoder_path = os.path.join(out_dir, "target_encoder.pt")
predictor_path = os.path.join(out_dir, "predictor.pt")
optimizer_path = os.path.join(out_dir, "optimizer.pt")
train_run_state_path = os.path.join(out_dir, "train_run_state.pt")

# train paths
train_context_encoder_path = os.path.join(out_dir, "train", "context_encoder.pt")
train_target_encoder_path = os.path.join(out_dir, "train", "target_encoder.pt")
train_predictor_path = os.path.join(out_dir, "train", "predictor.pt")
train_optimizer_path = os.path.join(out_dir, "train", "optimizer.pt")
train_train_run_state_path = os.path.join(out_dir, "train", "train_run_state.pt")



encoder_config_path = os.path.join(out_dir, "encoder_config.json")
predictor_config_path = os.path.join(out_dir, "predictor_config.json")
opt_config_path = os.path.join(out_dir, "opt_config.json")
train_run_config_path = os.path.join(out_dir, "train_run_config.json")

#%%
max_context_mask_ratio = 0.8#1.0 # how much of the input should be included in the context
max_target_mask_ratio = .25# how much of the input should be used for targets
target_block_num = 4

batch_size = train_run_config.batch_size
accumulation_steps = train_run_config.accumulation_steps
eval_interval = train_run_config.eval_interval
num_eval_batches = train_run_config.num_eval_batches

random_seed = train_run_config.random_seed

grad_clip = 1.0


#%%
if init_from == "scratch":
    set_seed(random_seed)

    context_encoder = Encoder(encoder_config) # initialize context encoder
    target_encoder = copy.deepcopy(context_encoder) # create target encoder as a copy of context encoder
    # freeze target encoder
    for param in target_encoder.parameters():
        param.requires_grad = False

    predictor = Predictor(predictor_config) # initialize predictor

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(train_out_dir, exist_ok=True)


    dataclass_to_json(encoder_config, encoder_config_path)
    dataclass_to_json(predictor_config, predictor_config_path)
    dataclass_to_json(opt_config, opt_config_path)
    dataclass_to_json(train_run_config, train_run_config_path)
    
elif init_from == "resume":
    encoder_config = json_to_dataclass(EncoderConfig, encoder_config_path)
    predictor_config = json_to_dataclass(PredictorConfig, predictor_config_path)
    opt_config = json_to_dataclass(OptimizerConfig, opt_config_path)
    train_run_config = json_to_dataclass(TrainRunConfig, train_run_config_path)

    context_encoder = Encoder(encoder_config)
    target_encoder = Encoder(encoder_config)
    predictor = Predictor(predictor_config)

    resume_context_encoder_path = train_context_encoder_path if resume_from == "train" else context_encoder_path
    resume_target_encoder_path = train_target_encoder_path if resume_from == "train" else target_encoder_path
    resume_predictor_path = train_predictor_path if resume_from == "train" else predictor_path
    resume_train_run_state_path = train_train_run_state_path if resume_from == "train" else train_run_state_path
    

    context_encoder.load_state_dict(torch.load(resume_context_encoder_path))
    target_encoder.load_state_dict(torch.load(resume_target_encoder_path))
    predictor.load_state_dict(torch.load(resume_predictor_path))

    train_run_data = torch.load(resume_train_run_state_path)

    # freeze target encoder
    for param in target_encoder.parameters():
        param.requires_grad = False

#%%
if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config={
        'encoder_config': encoder_config.__dict__ | {"n_params": context_encoder.get_num_params()},
        'predictor_config': predictor_config.__dict__ | {"n_params": predictor.get_num_params()},
        'opt_config': opt_config.__dict__,
        'train_run_config': train_run_config.__dict__
    })

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

#%%
tokenizer = LlamaTokenizer.from_pretrained('llama_tokenizer', use_fast = False) # initialize tokenizer
tokenizer.pad_token = tokenizer.eos_token

#%%
#TODO: move to prep data script 
dataset = load_from_disk('data/TinyStories').filter(lambda x: tokenizer(x['text'], return_tensors="pt", add_special_tokens = False)['input_ids'].shape[1] > 8, num_proc=12)

#%%


# init momentum scheduler
ema = opt_config.ema
bipe_scale = opt_config.bipe_scale
batch_iterations_per_epoch = math.ceil(len(dataset['train']) / (batch_size * accumulation_steps)) # TODO: add .floor if the last batch should be dropped
num_epochs = opt_config.num_epochs
final_lr = opt_config.final_lr
final_wd = opt_config.final_weight_decay
lr = opt_config.lr
start_lr = opt_config.start_lr
warmup = opt_config.warmup
wd = opt_config.weight_decay

#%%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_type = 'cuda' if 'cuda' in device else 'cpu'

assert device_type == 'cuda', 'CPU training is not supported'

dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
type_casting = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

#%%
# load the models to the device
context_encoder.to(device)
target_encoder.to(device)
predictor.to(device)

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

max_iter_num = math.ceil(len(dataset["train"]) / batch_size) if not max_iter_num else max_iter_num
iter_num = 0 if init_from == "scratch" else train_run_data['iter_num'] + 1
assert iter_num % accumulation_steps == 0, 'iter_num must be divisible by accumulation_steps without remainder. May be loaded incorrectly from resume dir'
assert eval_interval % accumulation_steps == 0, 'eval_interval must be divisible by accumulation_steps without remainder'
weight_update_iter_num = iter_num // accumulation_steps

optimizer = torch.optim.AdamW(param_groups)
if init_from == "resume":
    resume_optimizer_path = train_optimizer_path if resume_from == "train" else optimizer_path
    optimizer.load_state_dict(torch.load(resume_optimizer_path))

lr_scheduler = WarmupCosineSchedule(
    optimizer,
    warmup_steps=int(warmup*batch_iterations_per_epoch),
    start_lr=start_lr,
    ref_lr=lr,
    final_lr=final_lr,
    T_max=int(bipe_scale*num_epochs*batch_iterations_per_epoch),
    step=weight_update_iter_num
)

wd_scheduler = CosineWDSchedule(
    optimizer,
    ref_wd=wd,
    final_wd=final_wd,
    T_max=int(bipe_scale*num_epochs*batch_iterations_per_epoch),
    step=weight_update_iter_num
)

ema_scheduler = ExponentialMovingAverageSchedule(
    momentum=ema[0],
    T_max=int(bipe_scale*num_epochs*batch_iterations_per_epoch),
    step=weight_update_iter_num
)


#%%
def get_batch(split, index, batch_size):
    data = dataset[split]
    return data[index:index+batch_size]

#%%
def compute_jepa_loss(
        input_ids, 
        input_attn_mask, 
        target_encoder, 
        context_encoder, 
        predictor, 
        max_target_mask_ratio = 0.5, 
        max_context_mask_ratio = 0.5, 
        target_block_num = 3, 
        device = 'cpu'
    ):
    """
    
    Parameters:
    input_ids: torch.tensor of shape (batch_size, input_length) containing the (padded) input ids
    input_attn_mask: torch.tensor of shape (batch_size, input_length) containing the input attention mask. 0s represent padding, 1s represent allowed inputs.
    target_encoder: the target encoder
    context_encoder: the context encoder
    predictor: the predictor
    max_target_mask_ratio: the maximum ratio of the input that can be used for targets
    max_context_mask_ratio: the maximum ratio of the input that can be used for context
    target_block_num: the number of target blocks to predict
    device: the device to use for computation

    Returns:
    loss: the loss of the prediction
    """

    
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
    for batch_index in range(batch_count):
        # if the target block size is smaller than the max target block size, then the indices past the target block size will not be updated (they are handled by the attention mask)
        target_block_indices[batch_index, :, :target_block_sizes[batch_index]] += torch.arange(target_block_sizes[batch_index]).view(1, -1) # add the indices of the target blocks to the tensor
    
    # for target packing we flatten the target block indices to the shape of (batch_size, target_block_num * block_size)
    target_block_indices = target_block_indices.view(batch_count, -1)
    # get the target blocks embeddings in the shape of (batch_size, target_block_num, block_size, embedding_size)
    target_blocks_embeddings = torch.stack([target_embeddings[index,target_block_range,:] for index, target_block_range in enumerate(target_block_indices)]) # get the target embeddings

    # make sure the target blocks embeddings are correctly selected
    for batch_index, sample_range in enumerate(target_block_indices):
        for jndex, _range in enumerate(sample_range):
            assert torch.all(target_embeddings[batch_index, _range, :] == target_blocks_embeddings[batch_index, jndex, :]) # make sure the target blocks embeddings are correctly selected
    # target embeddings are arranged in 
    # ((batch_sample_0, target_block_0 + ... + target_block_n, embedding_size),
    #  ...
    # (batch_sample_n, target_block_0 + ... + target_block_n, embedding_size) 
    # )
            
    # create boolean mask of allowed inputs in the context
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

    for batch_index, allowed_context_length in enumerate(max_allowed_context_lengths): # for each sample
        context_block_indices = torch.arange(0, input_length)[allowed_in_context[batch_index]] # get the indices of the context inputs
        perm = torch.randperm(context_block_indices.size(0)) # shuffle the indices
        idx = perm[:allowed_context_length] # select indices up to the smallest context length
        # context_blocks_indices.append(context_block_indices[idx]) # add the indices to the list
        context_blocks_indices[batch_index, :allowed_context_length] = context_block_indices[idx] # set the context blocks indices
        context_inputs[batch_index, :allowed_context_length] = input_ids[batch_index, context_block_indices[idx]] # set the context inputs
        context_attention_mask[batch_index, :allowed_context_length] = True # set the attention mask to true for the context inputs

    # get the context blocks embeddings
    context_blocks_embeddings = context_encoder(context_inputs.to(device), id_indices=context_blocks_indices.to(device), attn_mask=context_attention_mask.unsqueeze(1).unsqueeze(1).to(device))

    # create the prediction attention mask, which is the maximum context length + the maximum target block size * the number of target blocks
    prediction_input_size = max_context_length + max_target_block_size * target_block_num 
    # add extra 0s to the context attention mask representing the target input attentions 
    pred_context_attn_mask = torch.cat((context_attention_mask, torch.zeros((batch_count, max_target_block_size * target_block_num), dtype = torch.bool)), dim = 1)

    # repeat the context attention mask for the prediction input size. We are doing it this way so that the target inputs only attend to the context inputs but no paddings
    prediction_attn_mask = pred_context_attn_mask[:, None, :] * pred_context_attn_mask[:, :, None]

    # let the inputs in each target block attend to each other but not the other target blocks
    for batch_index, block_size in enumerate(target_block_sizes):
        #for each target block, add the target attention mask to the prediction attention mask
        for target_block_index in range(target_block_num):
            # create the predictor target attention mask of the predictor's input size
            pred_target_attn_mask = torch.zeros((prediction_input_size), dtype = torch.bool)
            target_attn_start_index = max_context_length + target_block_index * max_target_block_size
            pred_target_attn_mask[target_attn_start_index:target_attn_start_index+block_size] = True
            # 1. multiply the context attention mask by the target attention mask to have the target inputs attend to the context inputs
            # 2. multiply the target attention mask by itself to have the target inputs attend to each other
            prediction_attn_mask[batch_index] += pred_context_attn_mask[batch_index, None, :] * pred_target_attn_mask[:, None] +  pred_target_attn_mask[None, :] * pred_target_attn_mask[:, None]

    # if context inputs do not attend to atleast one other input, even if they are padding inputs, the transformer models will return nan values. We set the padding inputs to attend the context
    for i in range(prediction_attn_mask.shape[0]):
        ctx_len = torch.sum(prediction_attn_mask[i], dim = 1)[0]
        prediction_attn_mask[i, :, :ctx_len] = True

    # predict target blocks from context blocks
    mask_token_embedding = predictor.get_mask_token_embedding()
    # mask_toke_embeddings are the same so we can just repeat them, the will only be differentiated by the position embeddings
    prediction_embeddings = mask_token_embedding.repeat(batch_count, max_target_block_size * target_block_num, 1) # for the batch size, for the largest target block * number of targets per block, repeat the mask token 

    input_embeddings = torch.cat((context_blocks_embeddings, prediction_embeddings.to(device)), dim = 1) # concatenate the context and prediction embeddings

    input_indices = torch.cat((context_blocks_indices, target_block_indices), dim = 1) # concatenate the context and target indices

    predicted_embeddings = predictor(input_embeddings, input_indices.to(device), attn_mask=prediction_attn_mask.unsqueeze(1).to(device)) # predict the target embeddings

    _, context_length, _ = context_blocks_embeddings.shape
    predicted_target_embeddings = predicted_embeddings[:,context_length:] # remove the context predictions

    # only attend to the embeddings of the predicted target blocks
    relevant_target_attn_mask = torch.diagonal(prediction_attn_mask, dim1=1, dim2=2)
    relevant_target_attn_mask = relevant_target_attn_mask[:, context_length:].unsqueeze(2).to(device)

    # compute the loss of the masked predictions
    loss = F.smooth_l1_loss(predicted_target_embeddings * relevant_target_attn_mask, target_blocks_embeddings.view(predicted_target_embeddings.shape) * relevant_target_attn_mask) # compute the loss

    return loss

#%%
best_loss = 1e9
while iter_num < max_iter_num:
    epoch = iter_num // len(dataset["train"])
    
    set_seed(random_seed) #TODO: find better solution for reproducibility?
    batch_idx = iter_num % math.ceil(len(dataset["train"]) / batch_size)
    batch = get_batch('train', batch_idx, min(batch_size, len(dataset["train"]) - batch_idx * batch_size))
    
    with open(os.path.join(out_dir, 'batch.jsonl'), 'a') as f:
        f.write(json.dumps({'text': batch['text']}) + '\n')

    tokenized = tokenizer(batch['text'], padding = True, truncation = False, max_length = 1024, return_tensors="pt", add_special_tokens = False)
    with type_casting:
        loss = compute_jepa_loss(
            tokenized['input_ids'], 
            tokenized['attention_mask'], 
            target_encoder, 
            context_encoder, 
            predictor, 
            max_target_mask_ratio = max_target_mask_ratio, 
            max_context_mask_ratio = max_context_mask_ratio, 
            target_block_num = target_block_num, 
            device = device
        )

    assert not torch.isnan(loss), 'loss is nan!'

    loss /= accumulation_steps
    scaler.scale(loss).backward()

    # if the a full batch has been accumulated, update the model weights
    if (iter_num + 1) % accumulation_steps == 0:

        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(context_encoder.parameters(), grad_clip)
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad(set_to_none=True)

        _new_lr = lr_scheduler.step()
        _new_wd = wd_scheduler.step()
        _new_m = ema_scheduler.step(context_encoder, target_encoder)

        train_loss = loss.item()
        torch.save(context_encoder.state_dict(), train_context_encoder_path)
        torch.save(target_encoder.state_dict(), train_target_encoder_path)
        torch.save(predictor.state_dict(), train_predictor_path)
        torch.save(optimizer.state_dict(), train_optimizer_path)

        train_run_state = {
                'iter_num': iter_num,
                'epoch': epoch,
                'batch_idx': batch_idx,
                'loss': train_loss,
                'batch_size': batch_size,
                'torch_seed': torch.initial_seed(),
                'lr': lr
            }

        torch.save(train_run_state, train_train_run_state_path)

        if wandb_log:
            wandb.log({
                'train/loss': train_loss,
                'lr': _new_lr,
                'wd': _new_wd,
                'm': _new_m,
                # 'iter_num': iter_num
            }
            , step=iter_num)

        with open(os.path.join(out_dir, 'losses.jsonl'), 'a') as f:
            f.write(json.dumps({'loss': train_loss, 'iter_num' : iter_num}) + '\n')

    # if the eval interval has been reached, evaluate the model
    if iter_num % eval_interval == 0 and iter_num > 0:
        set_seed(random_seed + iter_num)
        
        context_encoder.eval()
        predictor.eval()

        mean_eval_loss = 0
        with torch.no_grad():
            for eval_iter in range(num_eval_batches):
                batch_idx = eval_iter % math.ceil(len(dataset["validation"]) / batch_size)
                batch = get_batch('validation', batch_idx, min(batch_size, len(dataset["validation"]) - batch_idx * batch_size))

                tokenized = tokenizer(batch['text'], padding = True, truncation = False, max_length = 1024, return_tensors="pt", add_special_tokens = False)
                with type_casting:
                    eval_loss = compute_jepa_loss(
                        tokenized['input_ids'], 
                        tokenized['attention_mask'], 
                        target_encoder, 
                        context_encoder, 
                        predictor, 
                        max_target_mask_ratio = max_target_mask_ratio, 
                        max_context_mask_ratio = max_context_mask_ratio, 
                        target_block_num = target_block_num, 
                        device = device
                    )

                mean_eval_loss += eval_loss.item() / num_eval_batches


        if mean_eval_loss < best_loss:
            best_loss = mean_eval_loss
            torch.save(context_encoder.state_dict(), context_encoder_path)
            torch.save(target_encoder.state_dict(), target_encoder_path)
            torch.save(predictor.state_dict(), predictor_path)
            torch.save(optimizer.state_dict(), optimizer_path)

            train_run_state = {
                    'iter_num': iter_num,
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                    'loss': mean_eval_loss,
                    'batch_size': batch_size,
                    'torch_seed': torch.initial_seed(),
                    'lr': lr
                }

            torch.save(train_run_state, train_run_state_path)


        if wandb_log:
            wandb.log({
                # 'train/loss': loss.item(),
                'val/loss': mean_eval_loss,
                # 'lr': _new_lr,
                # 'wd': _new_wd,
                # 'm': _new_m,
                # 'iter_num': iter_num
            }
            , step=iter_num)

        context_encoder.train()
        predictor.train()

    iter_num += 1


#%%
