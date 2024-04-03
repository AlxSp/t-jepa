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
import sys
import numpy as np
from torch.utils.data import DataLoader
from contextlib import nullcontext
from schedulers import CosineWDSchedule, ExponentialMovingAverageSchedule, WarmupCosineSchedule
from models import Encoder, EncoderConfig, LinearClassifierProbe, LinearClassifierProbeConfig
from transformers import LlamaTokenizer
from datasets import load_dataset
from tqdm import tqdm

import torch
import torch.nn.functional as F

#%%
parser = ArgumentParser(description='')
parser.add_argument('--init_from', type=str, required=False, choices=['scratch', 'resume'], help='init from scratch or resume')
parser.add_argument('--encoder_config_path', type=str, required=False, help='path to the encoder config')
# parser.add_argument('--predictor_config_path', type=str, required=False, help='path to the predictor config')
parser.add_argument('--classifier_config_path', type=str, required=False, help='path to the classifier config')
parser.add_argument('--opt_config_path', type=str, required=False, help='path to the optimizer config')
parser.add_argument('--train_run_config_path', type=str, required=False, help='path to the train run config')
# parser.add_argument('--target_masking_strategies_path', type=str, required=False, help='path to the target masking strategies')
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
    rotary_n_embd = 32,
    dropout = 0.0,
    bias = True, # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better on small datasets
) if not args.encoder_config_path else toml_to_dataclass(EncoderConfig, args.encoder_config_path)


classifier_config = LinearClassifierProbeConfig(
    n_embd = 384,
    n_classes = 2,
)

@dataclass
class TrainRunConfig:
    batch_size: int = 40
    accumulation_steps: int = 4
    eval_interval: int = 40
    num_eval_batches: int = 20
    max_iter_num: int | None = None
    random_seed: int = 42

train_run_config = TrainRunConfig(
    batch_size = 32,
    accumulation_steps=64,
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
    lr: float = 0.0001
    start_lr: float = 0.00002
    final_lr: float = 1.0e-06
    grad_clip: float = 1.0

#%%
opt_config = OptimizerConfig(
    num_epochs = 100,
    ema = (0.996, 1.0),
    bipe_scale = 1.0, # batch iterations per epoch scale
    weight_decay = 0.04,
    final_weight_decay = 0.4,
    warmup_steps = 0.025,
    lr = 0.001, # 0.001
    start_lr = 0.0002,
    final_lr = 1.0e-06,
) if not args.opt_config_path else toml_to_dataclass(OptimizerConfig, args.opt_config_path)

#TODO: remove
context_max_mask_ratio = 0.8#1.0 # how much of the input should be included in the context
target_max_mask_ratio = .25# how much of the input should be used for targets
target_block_max_num = 4
target_block_min_num = 1 


#%%
dataset_name = "sst2"
dataset_dir = os.path.join("data", dataset_name)

max_input_length = 1024
#%%
wandb_log = True
wandb_project = "t-jepa-sentiment-probing"
wandb_run_name = "sent" #-1_epoch

# compile_model = True 

init_from = "resume"
init_from = "scratch"
init_from = init_from if not args.init_from else args.init_from
resume_from = "train" # "train" or "best"

print(f"init from: {init_from}")

out_dir = "classifier_out" if not args.output_dir else args.output_dir
train_out_dir = os.path.join(out_dir, "train")

max_iter_num = None

# best eval paths
context_encoder_path = os.path.join(out_dir, "context_encoder.pt")
target_encoder_path = os.path.join(out_dir, "target_encoder.pt")
# predictor_path = os.path.join(out_dir, "predictor.pt")

classifier_path = os.path.join(out_dir, "classifier.pt")
optimizer_path = os.path.join(out_dir, "optimizer.pt")
train_run_state_path = os.path.join(out_dir, "train_run_state.pt")

# train paths
train_context_encoder_path = os.path.join(out_dir, "train", "context_encoder.pt")
# train_target_encoder_path = os.path.join(out_dir, "train", "target_encoder.pt")
# train_predictor_path = os.path.join(out_dir, "train", "predictor.pt")
train_classifier_path = os.path.join(out_dir, "train", "classifier.pt")
train_optimizer_path = os.path.join(out_dir, "train", "optimizer.pt")
train_train_run_state_path = os.path.join(out_dir, "train", "train_run_state.pt")



encoder_config_path = os.path.join(out_dir, "encoder_config.json")
classifier_config_path = os.path.join(out_dir, "classifier_config.json")
# predictor_config_path = os.path.join(out_dir, "predictor_config.json")
opt_config_path = os.path.join(out_dir, "opt_config.json")
train_run_config_path = os.path.join(out_dir, "train_run_config.json")
# target_masking_strategies_path = os.path.join(out_dir, "target_masking_strategies.json")

#%%


#%%
batch_size = train_run_config.batch_size
accumulation_steps = train_run_config.accumulation_steps

random_seed = train_run_config.random_seed

grad_clip = 1.0


trained_context_encoder_path = os.path.join("out", "train", "context_encoder.pt")
# trained_predictor_path = os.path.join("out", "train", "predictor.pt")

#%%
if init_from == "scratch":
    set_seed(random_seed)

    context_encoder = Encoder(encoder_config) # initialize context encoder
    
    #FIXME: add path from original training
    context_encoder.load_state_dict(torch.load(trained_context_encoder_path))


    # freeze context and predictor
    for param in context_encoder.parameters():
        param.requires_grad = False

    for param in context_encoder.parameters():
        param.requires_grad = False

    classifier = LinearClassifierProbe(classifier_config) # initialize classifier

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(train_out_dir, exist_ok=True)

    torch.save(context_encoder.state_dict(), context_encoder_path)

    dataclass_to_json(encoder_config, encoder_config_path)
    dataclass_to_json(classifier_config, classifier_config_path)
    dataclass_to_json(opt_config, opt_config_path)
    dataclass_to_json(train_run_config, train_run_config_path)

    
elif init_from == "resume":
    encoder_config = json_to_dataclass(EncoderConfig, encoder_config_path)
    # predictor_config = json_to_dataclass(PredictorConfig, predictor_config_path)
    opt_config = json_to_dataclass(OptimizerConfig, opt_config_path)
    train_run_config = json_to_dataclass(TrainRunConfig, train_run_config_path)
    # target_masking_strategies = json_to_dict(target_masking_strategies_path)


    context_encoder = Encoder(encoder_config)
    # predictor = Predictor(predictor_config)
    classifier = LinearClassifierProbe(classifier_config)

    # resume_context_encoder_path = train_context_encoder_path if resume_from == "train" else context_encoder_path
    # resume_predictor_path = train_predictor_path if resume_from == "train" else predictor_path
    resume_train_run_state_path = train_train_run_state_path if resume_from == "train" else train_run_state_path
    

    context_encoder.load_state_dict(torch.load(context_encoder_path))
    # predictor.load_state_dict(torch.load(predictor_path))

    train_run_data = torch.load(resume_train_run_state_path)

    # freeze context and predictor
    for param in context_encoder.parameters():
        param.requires_grad = False

    for param in context_encoder.parameters():
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
            # 'predictor_config': predictor_config.__dict__ | {"n_params": predictor.get_num_params()},
            'opt_config': opt_config.__dict__,
            'train_run_config': train_run_config.__dict__,
            # 'target_masking_strategies': target_masking_strategies,
        },
        resume=True if init_from == "resume" else False
        )

#%%
tokenizer = LlamaTokenizer.from_pretrained('llama_tokenizer', use_fast = False) # initialize tokenizer
tokenizer.pad_token = tokenizer.eos_token

#%%
#TODO: move to prep data script 
# dataset = load_from_disk(dataset_dir)

dataset = load_dataset(dataset_name, cache_dir=dataset_dir, num_proc=12)

train_set_len = len(dataset['train'])
val_set_len = len(dataset['validation'])

#%%
eval_interval = train_run_config.eval_interval
num_eval_batches = math.ceil(val_set_len / (batch_size))

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
# predictor.to(device)
classifier.to(device)

#%%



max_iter_num = math.ceil(train_set_len * num_epochs / batch_size) if not max_iter_num else max_iter_num
iter_num = 0 if init_from == "scratch" else train_run_data['iter_num'] + 1
assert iter_num % accumulation_steps == 0, 'iter_num must be divisible by accumulation_steps without remainder. May be loaded incorrectly from resume dir'
assert eval_interval % accumulation_steps == 0, 'eval_interval must be divisible by accumulation_steps without remainder'
weight_update_iter_num = iter_num // accumulation_steps

param_groups = [
        {
            'params': (p for n, p in classifier.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        },{
            'params': (p for n, p in classifier.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }
    ]

optimizer = torch.optim.AdamW(param_groups, lr=start_lr)
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

# ema_scheduler = ExponentialMovingAverageSchedule(
#     momentum=ema[0],
#     T_max=int(bipe_scale*num_epochs*batch_iterations_per_epoch),
#     step=weight_update_iter_num
# )

# FIXME: add support for compiled models, currently causes an error
# if compile_model:
#     target_encoder = torch.compile(target_encoder)
#     context_encoder = torch.compile(context_encoder)
#     predictor = torch.compile(predictor)

#%%
def get_batch(split, index, batch_size, tokenizer, max_length):
    #TODO: get labels
    data = dataset[split]
    samples = data[index:index+batch_size]

    tokenized = tokenizer(samples['sentence'], padding = True, truncation = True, max_length = max_length, return_tensors="pt", add_special_tokens = False)

    input_ids, attention_mask = tokenized['input_ids'], tokenized['attention_mask']

    labels = torch.tensor(samples['label'], dtype=torch.long)

    if device_type == 'cuda':
        input_ids = input_ids.pin_memory().to(device, non_blocking=True)
        attention_mask = attention_mask.pin_memory().to(device, non_blocking=True)
        labels = labels.pin_memory().to(device, non_blocking=True)
    else:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

    #FIXME: make sure that no 0 length inputs are left after the prepare script ran
    # filter out 0 length inputs
    lengths = torch.sum(attention_mask, dim = 1)
    mask = lengths > 0
    input_ids = input_ids[mask]
    attention_mask = attention_mask[mask]
    labels = labels[mask]
    return input_ids, attention_mask, labels

#%%
def mean_pooling(embeddings, attention_mask):
    # embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    return torch.sum(embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

#%%
total_inputs_seen = 0 if init_from == "scratch" else train_run_data['total_inputs_seen']
best_loss = 1e9
pbar = tqdm(total=max_iter_num - iter_num)
while iter_num < max_iter_num:
    epoch = iter_num // train_set_len
    
    set_seed(random_seed) #TODO: find better solution for reproducibility?
    batch_idx = (iter_num % math.ceil(train_set_len / batch_size)) * batch_size 
    input_ids, attention_mask, labels = get_batch('train', batch_idx, min(batch_size, train_set_len - batch_idx), tokenizer, max_input_length)
    
    # with open(os.path.join(out_dir, 'batch.jsonl'), 'a') as f:
    #     f.write(json.dumps({'text': batch['text']}) + '\n')
    with type_casting:
        with torch.no_grad():
            embeddings = context_encoder(input_ids, attn_mask = attention_mask.unsqueeze(1).unsqueeze(1).bool()) # get target embeddings, no need to provide input indices.
        #     target_embeddings = F.layer_norm(target_embeddings, (target_embeddings.shape[-1],)) # normalize the target embeddings
            
            embeddings = mean_pooling(embeddings, attention_mask)

        logits = classifier(embeddings)

        loss = F.cross_entropy(logits, labels)

    assert not torch.isnan(loss), 'loss is nan!'

    loss /= accumulation_steps
    scaler.scale(loss).backward()
    train_loss = loss.item()

    # total_inputs_seen += sum(true_input_lengths)

    # if the a full batch has been accumulated, update the model weights
    if (iter_num + 1) % accumulation_steps == 0:

        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad(set_to_none=True)

        _new_lr = lr_scheduler.step()
        _new_wd = wd_scheduler.step()
        # _new_m = ema_scheduler.step(context_encoder, target_encoder)

        # train_loss = loss.item()
        # torch.save(context_encoder.state_dict(), train_context_encoder_path)
        # torch.save(target_encoder.state_dict(), train_target_encoder_path)
        # torch.save(predictor.state_dict(), train_predictor_path)
        torch.save(classifier.state_dict(), train_classifier_path)
        torch.save(optimizer.state_dict(), train_optimizer_path)

        train_run_state = {
                'iter_num': iter_num,
                # 'total_inputs_seen' : total_inputs_seen,
                'epoch': epoch,
                'batch_idx': batch_idx,
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
                # 'm': _new_m,
                # 'iter_num': iter_num
            }
            , step=iter_num * batch_size) #FIXME iter_num is not well defined, maybe use number of inputs seen?

        # with open(os.path.join(out_dir, 'losses.jsonl'), 'a') as f:
        #     f.write(json.dumps({'loss': train_loss, 'iter_num' : iter_num}) + '\n')

    # if the eval interval has been reached, evaluate the model
    if iter_num % eval_interval == 0 and iter_num > 0:
        set_seed(random_seed + iter_num)
        
        context_encoder.eval()
        classifier.eval()

        mean_eval_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for eval_iter in range(num_eval_batches):
            batch_idx = (eval_iter % math.ceil(val_set_len / batch_size)) * batch_size
            input_ids, attention_mask, labels = get_batch('validation', batch_idx, min(batch_size, train_set_len - batch_idx), tokenizer, max_input_length)

            with type_casting:

                with torch.no_grad():
                    embeddings = context_encoder(input_ids, attn_mask = attention_mask.unsqueeze(1).unsqueeze(1).bool()) # get target embeddings, no need to provide input indices.
                #     target_embeddings = F.layer_norm(target_embeddings, (target_embeddings.shape[-1],)) # normalize the target embeddings
                    
                    embeddings = mean_pooling(embeddings, attention_mask)

                logits = classifier(embeddings)

                loss = F.cross_entropy(logits, labels)


                assert not torch.isnan(loss), 'loss is nan!'

                eval_loss = loss.item()

                mean_eval_loss += eval_loss / num_eval_batches

                correct_predictions += (logits.argmax(dim=1) == labels).type(torch.float).sum().item()

                total_predictions += len(labels)

        if mean_eval_loss < best_loss:
            best_loss = mean_eval_loss
            # torch.save(context_encoder.state_dict(), context_encoder_path)
            # torch.save(target_encoder.state_dict(), target_encoder_path)
            # torch.save(predictor.state_dict(), predictor_path)
            torch.save(classifier.state_dict(), classifier_path)
            torch.save(optimizer.state_dict(), optimizer_path)

            train_run_state = {
                    'iter_num': iter_num,
                    'total_inputs_seen' : total_inputs_seen,
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                    'loss': mean_eval_loss,
                    'accuracy': correct_predictions / total_predictions,
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
                'val/accuracy': correct_predictions / total_predictions,
                # 'lr': _new_lr,
                # 'wd': _new_wd,
                # 'm': _new_m,
                # 'iter_num': iter_num
            }
            , step=iter_num * batch_size) #FIXME iter_num is not well defined, maybe use number of inputs seen?

        classifier.train()

    iter_num += 1
    pbar.update(1)

pbar.close()


#%%
