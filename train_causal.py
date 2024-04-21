#%%
import copy
from argparse import ArgumentParser
from dataclasses import dataclass
import math
import os
import json
import toml
import random
import shutil
import numpy as np
from torch.utils.data import DataLoader
from contextlib import nullcontext
from schedulers import CosineWDSchedule, ExponentialMovingAverageSchedule, WarmupCosineSchedule
from models import CausalEncoder, EncoderConfig, CausalPredictorWithMultiIndexing as Predictor, PredictorConfig
from transformers import LlamaTokenizer
from datasets import load_from_disk
from tqdm import tqdm

import torch
import torch.nn.functional as F

#%%
parser = ArgumentParser(description='')
parser.add_argument('--init_from', type=str, required=False, choices=['scratch', 'resume'], help='init from scratch or resume')
parser.add_argument('--encoder_config_path', type=str, required=False, help='path to the encoder config')
parser.add_argument('--predictor_config_path', type=str, required=False, help='path to the predictor config')
parser.add_argument('--opt_config_path', type=str, required=False, help='path to the optimizer config')
parser.add_argument('--train_run_config_path', type=str, required=False, help='path to the train run config')
parser.add_argument('--target_masking_strategies_path', type=str, required=False, help='path to the target masking strategies')
parser.add_argument('--output_dir', type=str, required=False, help='path to the output directory')
args = parser.parse_args()

#%%
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def dict_to_json(dictionary, path):
    with open(path, 'w') as f:
        json.dump(dictionary, f, indent=2)

def json_to_dict(path):
    with open(path, 'r') as f:
        return json.load(f)

def dataclass_to_json(dataclass, path):
    with open(path, 'w') as f:
        json.dump(dataclass.__dict__, f, indent=2)

def json_to_dataclass(dataclass, path):
    with open(path, 'r') as f:
        data = json.load(f)
    return dataclass(**data)

def toml_to_dataclass(dataclass, path):
    with open(path, 'r') as f:
        data = toml.load(f)
    return dataclass(**data)

#%%
encoder_config = EncoderConfig(
    block_size = 1024,
    vocab_size = 32000, # LLAMA tokenizer is 32000, which is a multiple of 64 for efficiency
    n_layer = 8,
    n_head = 12,
    n_embd = 384,
    rotary_n_embd = 16,
    dropout = 0.0,
    bias = True, # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better on small datasets
) if not args.encoder_config_path else toml_to_dataclass(EncoderConfig, args.encoder_config_path)


predictor_config = PredictorConfig(
    n_layer = 8,
    n_head = 12,
    ext_n_embd = 384,
    n_embd = 384,
    rotary_n_embd = 16,
    dropout = 0.0,
    bias = True, # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better on small datasets
    trainable_mask_emb=True#False
) if not args.predictor_config_path else toml_to_dataclass(PredictorConfig, args.predictor_config_path)

@dataclass
class TrainRunConfig:
    batch_size: int = 40
    accumulation_steps: int = 4
    eval_interval: int = 40
    num_eval_batches: int = 20
    max_iter_num: int | None = None
    random_seed: int = 42

train_run_config = TrainRunConfig(
    batch_size = 224,
    accumulation_steps=8,
    eval_interval=1024,
    num_eval_batches = 200,
    max_iter_num=None,
    random_seed=42
) if not args.train_run_config_path else toml_to_dataclass(TrainRunConfig, args.train_run_config_path)


@dataclass
class OptimizerConfig:
    num_epochs: int = 100
    ema: tuple[float, float] = (0.996, 1.0)
    bipe_scale: float = 1.0
    weight_decay: float = 0.04
    final_weight_decay: float = 0.4
    warmup_steps: int = 0.025
    lr: float = 0.001
    start_lr: float = 0.0002
    final_lr: float = 1.0e-06
    grad_clip: float = 1.0

#%%
opt_config = OptimizerConfig(
    num_epochs = 15,
    ema = (0.998, 1.0),
    bipe_scale = 1.025, # batch iterations per epoch scale
    weight_decay = 0.04,
    final_weight_decay = 0.4,
    warmup_steps = 3,
    lr = 0.000625, # 0.001
    start_lr = 0.0002,
    final_lr = 1.0e-06,
) if not args.opt_config_path else toml_to_dataclass(OptimizerConfig, args.opt_config_path)

#TODO: remove
default_context_max_mask_ratio = 0.8#1.0 # how much of the input should be included in the context
default_target_max_mask_ratio = .25# how much of the input should be used for targets
target_block_max_num = 4
target_block_min_num = 1 

#%%
# @dataclass
# class TargetMaskingStrategy:
#     target_block_size: int | None = 8
#     target_block_size: int | None = 8
#     target_block_size_std: float | None = 0.15
#     target_max_mask_ratio: float = 0.25
#     target_block_max_num: int | None = None
#     target_mask_start_ratio: float | None = 0.0
#     context_max_mask_ratio: float | None = 1.0

#%%
# new format
# target_masking_strategies = [
#     # R denoising
#     # {
#     #     "target_block_size" : 3,
#     #     # "target_block_size_std" : 0.15,
#     #     "target_max_mask_ratio" : 0.25,
#     #     "target_block_max_num" : None,
#     # },
#     {
#         "target_block_size" : 8,
#         # "target_block_size_std" : 0.15,
#         "target_max_mask_ratio" : 0.35,
#         "target_block_max_num" : None,
#     },
#     # X denoising
#     {
#         "target_block_size" : 3,
#         # "target_block_size_std" : 0.5,
#         "target_max_mask_ratio" : 0.5,
#         "target_block_max_num" : None,
#     },
#     # {
#     #     "target_block_size" : 8,
#     #     # "target_block_size_std" : 0.5,
#     #     "target_max_mask_ratio" : 0.8,
#     #     "target_block_max_num" : None,
#     # },
#     # {
#     #     "target_block_size" : 64,
#     #     # "target_block_size_std" : 0.5,
#     #     "target_max_mask_ratio" : 0.15,
#     # },
#     {
#         "target_block_size" : 64,
#         # "target_block_size_std" : 0.5,
#         "target_max_mask_ratio" : 0.35,
#     },
#     # S denoising
#     # {
#     #     "target_block_size" : None,
#     #     # "target_block_size_std" : 0.5,
#     #     "target_mask_start_ratio" : 0.75,
#     #     "target_max_mask_ratio" : 0.25,
#     #     "target_block_max_num" : 1,
#     #     "context_max_mask_ratio" : 1.0,
#     # }

# ]

# target_masking_strategies = target_masking_strategies if not args.target_masking_strategies_path else json_to_dict(args.target_masking_strategies_path)

#%%
dataset_name = "MiniPile"
dataset_dir = os.path.join("data", dataset_name)

max_input_length = 256
#%%
wandb_log = True
wandb_project = "t-jepa-MiniPile-256ctx-causal"
wandb_run_name = "multi-mean-v1" #-1_epoch

# compile_model = True 

init_from = "resume"
init_from = "scratch"
init_from = init_from if not args.init_from else args.init_from
resume_from = "train" # "train" or "best"

print(f"init from: {init_from}")

out_dir = "out" if not args.output_dir else args.output_dir
train_out_dir = os.path.join(out_dir, "train")

max_iter_num = None

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
target_masking_strategies_path = os.path.join(out_dir, "target_masking_strategies.json")

#%%


#%%
batch_size = train_run_config.batch_size
accumulation_steps = train_run_config.accumulation_steps
eval_interval = train_run_config.eval_interval
num_eval_batches = train_run_config.num_eval_batches

random_seed = train_run_config.random_seed

grad_clip = 10.0


#%%
if init_from == "scratch":
    set_seed(random_seed)

    context_encoder = CausalEncoder(encoder_config) # initialize context encoder
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
    # dict_to_json(target_masking_strategies, target_masking_strategies_path)
    # with open(target_masking_strategies_path, 'w') as f:
    #     json.dump(target_masking_strategies, f, indent=2)

    
elif init_from == "resume":
    encoder_config = json_to_dataclass(EncoderConfig, encoder_config_path)
    predictor_config = json_to_dataclass(PredictorConfig, predictor_config_path)
    opt_config = json_to_dataclass(OptimizerConfig, opt_config_path)
    train_run_config = json_to_dataclass(TrainRunConfig, train_run_config_path)
    # target_masking_strategies = json_to_dict(target_masking_strategies_path)


    context_encoder = CausalEncoder(encoder_config)
    target_encoder = CausalEncoder(encoder_config)
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
    wandb.init(
        project=wandb_project, 
        name=wandb_run_name, 
        config=
            {
            'encoder_config': encoder_config.__dict__ | {"n_params": context_encoder.get_num_params()},
            'predictor_config': predictor_config.__dict__ | {"n_params": predictor.get_num_params()},
            'opt_config': opt_config.__dict__,
            'train_run_config': train_run_config.__dict__,
            # 'target_masking_strategies': target_masking_strategies,
        },
        resume=True if init_from == "resume" else False
        )

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

#%%
tokenizer = LlamaTokenizer.from_pretrained('llama_tokenizer', use_fast = False) # initialize tokenizer
tokenizer.pad_token = tokenizer.eos_token

#%%
dataset = load_from_disk(dataset_dir)

train_set_len = len(dataset['train'])
val_set_len = len(dataset['validation'])

#%%


# init momentum scheduler
ema = opt_config.ema
bipe_scale = opt_config.bipe_scale
batch_iterations_per_epoch = math.ceil(train_set_len / (batch_size * accumulation_steps)) # TODO: add .floor if the last batch should be dropped
num_epochs = opt_config.num_epochs
final_lr = opt_config.final_lr
final_wd = opt_config.final_weight_decay
lr = opt_config.lr
start_lr = opt_config.start_lr
warmup_steps = opt_config.warmup_steps
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



max_iter_num = math.ceil(train_set_len / batch_size) * num_epochs if not max_iter_num else max_iter_num
iter_num = 0 if init_from == "scratch" else train_run_data['iter_num'] + 1
assert iter_num % accumulation_steps == 0, 'iter_num must be divisible by accumulation_steps without remainder. May be loaded incorrectly from resume dir'
assert eval_interval % accumulation_steps == 0, 'eval_interval must be divisible by accumulation_steps without remainder'
weight_update_iter_num = iter_num // accumulation_steps

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

optimizer = torch.optim.AdamW(param_groups, lr=start_lr, betas = (0.9, 0.95))
if init_from == "resume":
    resume_optimizer_path = train_optimizer_path if resume_from == "train" else optimizer_path
    optimizer.load_state_dict(torch.load(resume_optimizer_path))

lr_scheduler = WarmupCosineSchedule(
    optimizer,
    warmup_steps=int(warmup_steps*batch_iterations_per_epoch),
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

# FIXME: add support for compiled models, currently causes an error
# if compile_model:
#     target_encoder = torch.compile(target_encoder)
#     context_encoder = torch.compile(context_encoder)
#     predictor = torch.compile(predictor)


#%%
# def collate_jepa_input_data(input_attn_mask, true_input_lengths, target_max_mask_ratio, target_mask_start_ratio, target_block_nums, target_block_size, context_max_mask_ratio):
#     batch_count = true_input_lengths.shape[0]  # get the batch count
#     max_length = input_attn_mask.shape[1]  # get the maximum length of the input

    
#     target_block_max_num = np.ceil(np.max(target_block_nums)).astype(int)  # get the maximum number of target blocks
#     target_block_sizes = np.clip((true_input_lengths * target_max_mask_ratio // target_block_nums), 1, target_block_size).astype(int)  # compute the target block sizes for each sample
#     target_max_block_size = np.max(target_block_sizes)  # get the maximum target block size
#     target_min_start_indices = np.clip(np.ceil(true_input_lengths * target_mask_start_ratio).astype(int), 0, None)  # compute the minimum starting index for the target blocks
    
#     # we need to shrink the input value for randperm such that only n amount of possible start values can be selected
#     potential_start_indices_max = np.clip(true_input_lengths - target_min_start_indices - target_block_sizes, 1, None)

#     target_block_start_indices = np.stack(
#         [
#             np.tile(
#                 np.random.permutation(max_index), 
#                 (np.ceil(target_max_block_size * target_block_max_num / max_index).astype(int))
#             )[:target_block_max_num] 
#             for max_index in potential_start_indices_max
#         ]
#     ) + target_min_start_indices.reshape(-1, 1)
    
#     target_blocks_indices = np.ones((batch_count, target_block_max_num, target_max_block_size), dtype = np.int64) * target_block_start_indices.reshape((batch_count, target_block_max_num, 1))
#     # for each sample add the indices of the target blocks to the tensor by applying a range to each row of the index tensors
#     for batch_index in range(batch_count):
#         # if the target block size is smaller than the max target block size, then the indices past the target block size will not be updated (they are handled by the attention mask)
#         target_blocks_indices[batch_index, :, :target_block_sizes[batch_index]] += np.arange(target_block_sizes[batch_index]).reshape(1, -1) # add the indices of the target blocks to the tensor
#     # for target packing we flatten the target block indices to the shape of (batch_size, target_block_max_num * block_size)
#     target_blocks_indices = target_blocks_indices.reshape(batch_count, -1)
#     # target embeddings will be arranged in the following way: 
#     # ((batch_sample_0, target_block_0 + ... + target_block_n, embedding_size),
#     #  ...
#     # (batch_sample_n, target_block_0 + ... + target_block_n, embedding_size) 
#     # )
    
#     # create boolean mask of allowed inputs in the context
#     allowed_in_context = input_attn_mask.copy().astype(bool) # create a tensor of trues, representing the allowed inputs in the context
#     for batch_index, target_block_range in enumerate(target_blocks_indices):
#         # get the relevant target block range for each sample which correspond to the number of target blocks and the target block size
#         utilized_target_indices_range = (target_block_nums[batch_index] - 1) * target_max_block_size + target_block_sizes[batch_index]
#         relevant_target_block_range = target_block_range[: utilized_target_indices_range]
#         allowed_in_context[batch_index, relevant_target_block_range] = False
#     # make sure all context blocks have the same length
#     context_lengths = np.sum(allowed_in_context, axis=1)
#     assert np.all(context_lengths > 0), "All samples must have at least one input in the context"
#     max_allowed_context_lengths = np.min(np.array([context_lengths, np.clip(true_input_lengths * context_max_mask_ratio, 1, None).astype(int)]), axis=0) # compute the max allowed context lengths
#     max_context_length = np.max(max_allowed_context_lengths) # get the max allowed context length
    
#     context_blocks_indices = np.zeros((batch_count, max_context_length), dtype=np.int64) # create a tensor of zeros for the context blocks indices
#     context_attn_mask = np.zeros((batch_count, max_context_length), dtype=bool) # create a tensor of zeros for the context attention mask
    
#     for batch_index, allowed_context_length in enumerate(max_allowed_context_lengths): # for each sample
#         context_block_indices = np.arange(0, max_length)[allowed_in_context[batch_index]]  # get the indices of the context inputs
#         perm = np.random.permutation(len(context_block_indices))  # shuffle the indices
#         idx = perm[:allowed_context_length] # select indices up to the allowed context length for the sample
#         context_blocks_indices[batch_index, :allowed_context_length] = context_block_indices[idx]  # set the context blocks indices
#         context_attn_mask[batch_index, :allowed_context_length] = True  # set the attention mask to true for the context inputs

#     # concatenate the context and target indices to get the predictor input indices
#     predictor_input_indices = np.concatenate((context_blocks_indices, target_blocks_indices), axis=1)

#     prediction_input_size = max_context_length + target_max_block_size * target_block_max_num
#     # create the predictor attention mask
#     predictor_attn_mask = np.zeros((batch_count, prediction_input_size, prediction_input_size), dtype=bool) # pred_context_attn_mask[:, np.newaxis, :] * pred_context_attn_mask[:, :, np.newaxis]
#     for batch_index, block_size in enumerate(target_block_sizes):
#         # have all inputs attend the context
#         allowed_context_length = max_allowed_context_lengths[batch_index]
#         predictor_attn_mask[batch_index, :, :allowed_context_length] = True
#         for target_block_index in range(target_block_nums[batch_index]):
#             # have all inputs in a target block attend to each other (as well as the context)
#             target_attn_start_index = max_context_length + target_block_index * target_max_block_size
#             predictor_attn_mask[batch_index, target_attn_start_index:target_attn_start_index+block_size, target_attn_start_index:target_attn_start_index+block_size] = True

#     return target_blocks_indices, context_blocks_indices, context_attn_mask, predictor_input_indices, predictor_attn_mask

class ResumableSampler:
    def __init__(self, dataset_len, offset = 0, batch_size = 1):
        self.dataset_len = dataset_len
        self.iter = offset
        self.batch_size = batch_size
        self.batch_num = math.ceil(self.dataset_len / self.batch_size)

    def __len__(self):
        return self.batch_num
    
    def __iter__(self):
        while True:
            batch_idx = self.iter % self.batch_num * self.batch_size
            yield list(range(batch_idx, min(batch_idx + self.batch_size, self.dataset_len)))
            self.iter += 1

class TJEPABatch:
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        # self.target_masking_strategy_data_list = target_masking_strategy_data_list

    def pin_memory(self):
        self.input_ids = self.input_ids.pin_memory()
        self.attention_mask = self.attention_mask.pin_memory()
        # self.target_masking_strategy_data_list = [tuple(map(lambda x: x.pin_memory(), strategy_data)) for strategy_data in self.target_masking_strategy_data_list]

        return self
    
    def __iter__(self):
        return iter([self.input_ids, self.attention_mask])

class TJEPADataCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        # self.target_masking_strategies = target_masking_strategies
        self.max_length = max_length
        # self.context_max_mask_ratio = context_max_mask_ratio # how much of the input should be included in the context
        # self.target_max_mask_ratio = target_max_mask_ratio # how much of the input should be used for targets
        # self.target_block_max_num = target_block_max_num
        # self.target_block_min_num = target_block_min_num 

    def __call__(self, batch):
        tokenized = self.tokenizer([b['text'] for b in batch], padding = True, truncation = True, max_length = self.max_length, return_tensors="np", add_special_tokens = False)

        input_ids, attention_mask = tokenized['input_ids'], tokenized['attention_mask']

        # filter out 0 length inputs
        lengths = np.sum(attention_mask, axis = 1)
        mask = lengths > 0
        input_ids = input_ids[mask]
        attention_mask = attention_mask[mask]

        # true_input_lengths = np.sum(attention_mask, axis = 1)

        # target_masking_strategy_data_list = []
        # for strategy in self.target_masking_strategies:
        #     target_max_mask_ratio = strategy.get("target_max_mask_ratio", self.target_max_mask_ratio)
        #     context_max_mask_ratio = strategy.get("context_max_mask_ratio", self.context_max_mask_ratio)
        #     target_block_size = strategy.get("target_block_size", 1)
            
        #     target_block_nums = np.array([strategy.get("target_block_max_num")] * len(true_input_lengths)) if strategy.get("target_block_max_num") is not None else true_input_lengths * target_max_mask_ratio // strategy.get("target_block_size", 1)
        #     target_block_nums = np.where(target_block_nums < target_block_min_num, target_block_min_num, target_block_nums).astype(int)
        #     target_mask_start_ratio = strategy.get("target_mask_start_ratio", 0.0)

        #     target_masking_strategy_data_list.append(tuple(map(lambda x: torch.from_numpy(x), collate_jepa_input_data(attention_mask, true_input_lengths, target_max_mask_ratio, target_mask_start_ratio, target_block_nums, target_block_size, context_max_mask_ratio))))

        return TJEPABatch(torch.from_numpy(input_ids), torch.from_numpy(attention_mask))
    

#%%
def compute_jepa_loss(
        context_encoder, 
        predictor,
        input_ids, 
        target_embeddings, 
        attn_mask,
        window_size,
    ):

    batch_count = input_ids.shape[0]

    # target_embeddings = target_embeddings # get all the embeddings 

    # target_blocks_embeddings = torch.stack([target_embeddings[index,target_block_range,:] for index, target_block_range in enumerate(target_block_indices)]) # get the target embeddings

    # unfolded_embeddings = target_embeddings.unfold(1, 3, 1) # unfold the embeddings


    target_embeddings = target_embeddings.unfold(1, window_size, 1).mean(dim = 3) # average the embeddings

    target_embeddings = F.layer_norm(target_embeddings, (target_embeddings.shape[-1],)) # normalize the target embeddings


    # context_inputs = torch.gather(input_ids, 1, context_block_indices)

    # get the context blocks embeddings
    context_embeddings = context_encoder(input_ids) #, attn_mask=context_attn_mask.unsqueeze(1).unsqueeze(1))

    # # predict target blocks from context blocks
    mask_token_embedding = predictor.get_mask_token_embedding()
    # # mask_toke_embeddings are the same so we can just repeat them, the will only be differentiated by the position embeddings
    prediction_embeddings = mask_token_embedding.repeat(batch_count, 1, 1) # for the batch size, for the largest target block * number of targets per block, repeat the mask token 
    # concatenate the context and prediction embedding
    input_embeddings = torch.cat((prediction_embeddings, context_embeddings), dim = 1) # concatenate the context and prediction embeddings

    predicted_target_embeddings = predictor(input_embeddings, start_offset = 1, end_offset = window_size) #, attn_mask=prediction_attn_mask.unsqueeze(1).unsqueeze(1)) # predict the target embeddings

    # _, context_length, _ = context_blocks_embeddings.shape
    # predicted_target_embeddings = predicted_embeddings[:,context_length:] # remove the context predictions

    # # only attend to the embeddings of the predicted target blocks
    # relevant_target_attn_mask = torch.diagonal(prediction_attn_mask, dim1=1, dim2=2)
    # relevant_target_attn_mask = relevant_target_attn_mask[:, context_length:].unsqueeze(2)

    # since we are predicting the average of a window of "future" embeddings, the predictions of the last inputs have to be masked
    valid_target_count = target_embeddings.shape[1]# - predicted_target_embeddings.shape[1] - 1

    # compute the loss of the masked predictions
    embedding_losses = F.smooth_l1_loss(predicted_target_embeddings[:, :valid_target_count, :], target_embeddings, reduction='none') # compute the loss

    # mask the losses
    # TODO: validate that sliced attn_mask masks out the correct targets  
    masked_embedding_losses = embedding_losses * attn_mask[:, valid_target_count * -1:].unsqueeze(2)

    # compute the loss
    loss = torch.sum(masked_embedding_losses) / torch.sum(attn_mask)

    return loss    

#%%
collator = TJEPADataCollator(
    tokenizer, 
    # target_masking_strategies,
    max_input_length, 
    # default_context_max_mask_ratio,
    # default_target_max_mask_ratio,
    # target_block_max_num,
    # target_block_min_num,
)

train_sampler = ResumableSampler(
    train_set_len, 
    offset = iter_num, 
    batch_size = batch_size
)

train_loader = torch.utils.data.DataLoader(dataset['train'], batch_sampler = train_sampler, collate_fn = collator, num_workers = 6, pin_memory = True, shuffle = False)
train_loader_iter = iter(train_loader)

val_sampler = ResumableSampler(
    val_set_len,
    offset = 0,
    batch_size = batch_size
)
val_loader = torch.utils.data.DataLoader(dataset['validation'], batch_sampler = val_sampler, collate_fn = collator, num_workers = 6, pin_memory = True, shuffle = False)
val_loader_iter = iter(val_loader)

#%%
pred_window_sizes = [1, 4, 16, 32]
total_inputs_seen = 0 if init_from == "scratch" else train_run_data['total_inputs_seen']
best_loss = 1e9
pbar = tqdm(total=max_iter_num - iter_num, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}" + " [{elapsed}<{remaining}, {rate_noinv_fmt}]")
while iter_num < max_iter_num:
    epoch = iter_num // train_set_len

    input_ids, attention_mask = next(train_loader_iter)
    input_ids = input_ids.to(device, non_blocking=True)
    attention_mask = attention_mask.to(device, non_blocking=True)
    # strategy_count = len(strategies)
    # with open(os.path.join(out_dir, 'batch.jsonl'), 'a') as f:
    #     f.write(json.dumps({'text': batch['text']}) + '\n')
    with torch.no_grad(), type_casting:
        target_embeddings = target_encoder(input_ids) # get target embeddings, no need to provide input indices.
        target_embeddings = F.layer_norm(target_embeddings, (target_embeddings.shape[-1],)) # normalize the target embeddings


    train_loss = 0
    # for target_block_indices, context_block_indices, context_attention_mask, prediction_input_indices, prediction_attn_mask in strategies:
        
    # target_block_indices = target_block_indices.to(device, non_blocking=True)
    # context_block_indices = context_block_indices.to(device, non_blocking=True)
    # context_attention_mask = context_attention_mask.to(device, non_blocking=True)
    # prediction_input_indices = prediction_input_indices.to(device, non_blocking=True)
    # prediction_attn_mask = prediction_attn_mask.to(device, non_blocking=True)
    for pred_window_size in pred_window_sizes:

        with type_casting:
            loss = compute_jepa_loss(
                context_encoder, 
                predictor,
                input_ids, 
                target_embeddings, 
                attention_mask,
                pred_window_size
            )

        assert not torch.isnan(loss), 'loss is nan!'

        loss /= len(pred_window_sizes)
        train_loss += loss.item()
        loss /= accumulation_steps
        scaler.scale(loss).backward()

    true_input_lengths = torch.sum(attention_mask, dim = 1).to('cpu') # get the true input length for each sample
    total_inputs_seen += sum(true_input_lengths)

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

        # train_loss = loss.item()
        torch.save(context_encoder.state_dict(), train_context_encoder_path)
        torch.save(target_encoder.state_dict(), train_target_encoder_path)
        torch.save(predictor.state_dict(), train_predictor_path)
        torch.save(optimizer.state_dict(), train_optimizer_path)

        train_run_state = {
                'iter_num': iter_num,
                'total_inputs_seen' : total_inputs_seen,
                'epoch': epoch,
                # 'batch_idx': batch_idx,
                'loss': train_loss,
                'batch_size': batch_size,
                'accumulation_steps': accumulation_steps,
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
            , step=iter_num * batch_size) #FIXME iter_num is not well defined, maybe use number of inputs seen?

        # with open(os.path.join(out_dir, 'losses.jsonl'), 'a') as f:
        #     f.write(json.dumps({'loss': train_loss, 'iter_num' : iter_num}) + '\n')

    # if the eval interval has been reached, evaluate the model
    if iter_num % eval_interval == 0 and iter_num > 0:
        set_seed(random_seed + iter_num)
        
        context_encoder.eval()
        predictor.eval()

        mean_eval_loss = 0
        with torch.no_grad():
            eval_iter = 0
            # for input_ids, attention_mask, strategies in val_loader:
            for eval_iter in range(num_eval_batches):
                
                input_ids, attention_mask = next(val_loader_iter)
                input_ids = input_ids.to(device, non_blocking=True)
                attention_mask = attention_mask.to(device, non_blocking=True)
                # strategy_count = len(strategies)

                with type_casting:
                    target_embeddings = target_encoder(input_ids) # get target embeddings, no need to provide input indices.
                    target_embeddings = F.layer_norm(target_embeddings, (target_embeddings.shape[-1],)) # normalize the target embeddings
                    
                # true_input_lengths = torch.sum(attention_mask, dim = 1).to('cpu') # get the true input length for each sample

                eval_loss = 0
                # for target_block_indices, context_block_indices, context_attention_mask, prediction_input_indices, prediction_attn_mask in strategies:
                    
                #     target_block_indices = target_block_indices.to(device, non_blocking=True)
                #     context_block_indices = context_block_indices.to(device, non_blocking=True)
                #     context_attention_mask = context_attention_mask.to(device, non_blocking=True)
                #     prediction_input_indices = prediction_input_indices.to(device, non_blocking=True)
                #     prediction_attn_mask = prediction_attn_mask.to(device, non_blocking=True)
                for pred_window_size in pred_window_sizes:

                    with type_casting:
                        loss = compute_jepa_loss(
                            context_encoder, 
                            predictor,
                            input_ids, 
                            target_embeddings, 
                            attention_mask,
                            pred_window_size
                        )

                    assert not torch.isnan(loss), 'loss is nan!'
                # with type_casting:
                #     loss = compute_jepa_loss(
                #         context_encoder, 
                #         predictor,
                #         input_ids, 
                #         target_embeddings, 
                #         attention_mask
                #     )

                # assert not torch.isnan(loss), 'loss is nan!'
                    loss /= len(pred_window_sizes)
                    eval_loss += loss.item()

                mean_eval_loss += eval_loss / num_eval_batches

        if mean_eval_loss < best_loss:
            best_loss = mean_eval_loss
            torch.save(context_encoder.state_dict(), context_encoder_path)
            torch.save(target_encoder.state_dict(), target_encoder_path)
            torch.save(predictor.state_dict(), predictor_path)
            torch.save(optimizer.state_dict(), optimizer_path)

            train_run_state = {
                    'iter_num': iter_num,
                    'total_inputs_seen' : total_inputs_seen,
                    'epoch': epoch,
                    # 'batch_idx': batch_idx,
                    'loss': mean_eval_loss,
                    'batch_size': batch_size,
                    'accumulation_steps': accumulation_steps,
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
            , step=iter_num * batch_size) #FIXME iter_num is not well defined, maybe use number of inputs seen?

        context_encoder.train()
        predictor.train()

    iter_num += 1
    pbar.update(1)

pbar.close()


#%%
