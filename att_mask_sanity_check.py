#%%
import copy

from dataclasses import dataclass
from schedulers import CosineWDSchedule, WarmupCosineSchedule
from src.models.encoder import Encoder, EncoderConfig, Predictor, PredictorConfig
from transformers import LlamaTokenizer
from datasets import load_from_disk
from tqdm import tqdm

import torch
import torch.nn.functional as F

#%%

small_encoder_config = EncoderConfig(
    block_size = 1024,
    vocab_size = 32000, # LLAMA tokenizer is 32000, which is a multiple of 64 for efficiency
    n_layer = 8,
    n_head = 12,
    n_embd = 384,
    dropout = 0.0,
    bias = True, # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better on small datasets
)


#%%
tokenizer = LlamaTokenizer.from_pretrained('llama_tokenizer') # initialize tokenizer
tokenizer.pad_token = tokenizer.eos_token
context_encoder = Encoder(small_encoder_config) # initialize context encoder
context_encoder.eval()
#%%
batch_list = [
    # "The next day, Lily took the dog for a walk"
    "The next day, Lily took the dog for a",
    # "Hello, how are you?"
]
#TDOO: figure out why the input has to be at least 12 tokens for all the tests to pass
tokenized = tokenizer(batch_list, padding = True, truncation = False, max_length = 1024, return_tensors="pt")

print(tokenized['input_ids'].shape)

batch_count = tokenized['input_ids'].shape[0]
input_length = tokenized['input_ids'].shape[1]
# prediction_count = batch_count * target_block_num
#%%
test_a_inputs = tokenized['input_ids'][0][:torch.sum(tokenized['attention_mask'][0])].unsqueeze(0)
#%%
test_a_indices = torch.arange(0, len(test_a_inputs[0])).unsqueeze(0)
#%%
a_context_blocks_embeddings = context_encoder(test_a_inputs, id_indices=None, attn_mask=None)

#%%
a_2_context_blocks_embeddings = context_encoder(test_a_inputs, id_indices=test_a_indices, attn_mask=None)
assert torch.allclose(a_context_blocks_embeddings, a_2_context_blocks_embeddings)

#%%
a_3_context_blocks_embeddings = context_encoder(test_a_inputs, id_indices=test_a_indices, attn_mask=torch.ones((1, 1, len(test_a_inputs[0])), dtype = torch.bool))
assert torch.allclose(a_context_blocks_embeddings, a_3_context_blocks_embeddings)
#%%
test_b_inputs = torch.cat((test_a_inputs[0], test_a_inputs[0]), 0).unsqueeze(0)
test_b_indices = torch.cat((test_a_indices[0], test_a_indices[0]), 0).unsqueeze(0)

#%%
att_mask = torch.zeros_like(test_b_inputs, dtype = torch.bool)
att_mask[0,:len(test_a_inputs[0])] = True
att_mask = att_mask.unsqueeze(1).unsqueeze(1)

#%%
b_context_blocks_embeddings = context_encoder(test_b_inputs, id_indices=test_b_indices, attn_mask=att_mask)

#%%
assert torch.allclose(a_context_blocks_embeddings, b_context_blocks_embeddings[:, :len(test_a_inputs[0])])

#%%
print("All tests passed!")

#%%
fails = []
batch_count = 1
for i in range(1, 1024):
    input_ids = torch.tensor([torch.arange(0, i)])

    batch_count = input_ids.shape[0]
    input_length = input_ids.shape[1]
    # prediction_count = batch_count * target_block_num
    test_a_inputs = input_ids[0][:i].unsqueeze(0)
    test_a_indices = torch.arange(0, len(test_a_inputs[0])).unsqueeze(0)
    a_context_blocks_embeddings = context_encoder(test_a_inputs, id_indices=None, attn_mask=None)

    # a_2_context_blocks_embeddings = context_encoder(test_a_inputs, id_indices=test_a_indices, attn_mask=None)
    # assert torch.allclose(a_context_blocks_embeddings, a_2_context_blocks_embeddings)

    # a_3_context_blocks_embeddings = context_encoder(test_a_inputs, id_indices=test_a_indices, attn_mask=torch.ones((1, 1, len(test_a_inputs[0])), dtype = torch.bool))
    # assert torch.allclose(a_context_blocks_embeddings, a_3_context_blocks_embeddings)
    test_b_inputs = torch.cat((test_a_inputs[0], test_a_inputs[0]), 0).unsqueeze(0)
    test_b_indices = torch.cat((test_a_indices[0], test_a_indices[0]), 0).unsqueeze(0)

    att_mask = torch.zeros_like(test_b_inputs, dtype = torch.bool)
    att_mask[0,:len(test_a_inputs[0])] = True
    att_mask = att_mask.unsqueeze(1).unsqueeze(1)

    b_context_blocks_embeddings = context_encoder(test_b_inputs, id_indices=test_b_indices, attn_mask=att_mask)
    try:
        assert torch.allclose(a_context_blocks_embeddings, b_context_blocks_embeddings[:, :len(test_a_inputs[0])])
    except AssertionError:
        fails.append(i)

#%%
fails

#%%

#%%
fails = []
i = 11
batch_size = 4 
input_ids = torch.stack([torch.randint(0, 16000, (i,)) for _ in range(batch_size)])
batch_count = input_ids.shape[0]
input_length = input_ids.shape[1]


test_a_inputs = input_ids
test_a_indices = torch.stack([torch.arange(0, len(test_a_inputs[0]))]) # TODO: check if indices are correct
a_context_blocks_embeddings = context_encoder(test_a_inputs, id_indices=None, attn_mask=None)


test_b_inputs = torch.cat((test_a_inputs, torch.randint_like(test_a_inputs, 0, 16000)), 1)
test_b_indices = torch.cat((test_a_indices, test_a_indices), 1)


att_mask = torch.zeros_like(test_b_inputs, dtype = torch.bool)
att_mask[:,:len(test_a_inputs[0])] = True
att_mask = att_mask.unsqueeze(1).unsqueeze(1)

#%%
att_mask @ att_mask.T


#%%
# b_context_blocks_embeddings = context_encoder(test_b_inputs, id_indices=test_b_indices, attn_mask=att_mask)


# assert torch.allclose(a_context_blocks_embeddings, b_context_blocks_embeddings[:, :len(test_a_inputs[0])])

a_x = context_encoder.transformer.wte(test_a_inputs)
b_x = context_encoder.transformer.wte(test_b_inputs)

assert torch.allclose(a_x, b_x[:, :len(test_a_inputs[0])])
assert torch.equal(a_x, b_x[:, :len(test_a_inputs[0])])


a_x = context_encoder.transformer.drop(a_x)
b_x = context_encoder.transformer.drop(b_x)

assert torch.allclose(a_x, b_x[:, :len(test_a_inputs[0])])
assert torch.equal(a_x, b_x[:, :len(test_a_inputs[0])])



ln_a_x = context_encoder.transformer.h[0].ln_1(a_x)
ln_b_x = context_encoder.transformer.h[0].ln_1(b_x)

assert torch.allclose(ln_a_x, ln_b_x[:, :len(test_a_inputs[0])])
assert torch.equal(ln_a_x, ln_b_x[:, :len(test_a_inputs[0])])

n_embd = context_encoder.transformer.h[0].attn.n_embd
n_head = context_encoder.transformer.h[0].attn.n_head

a_B, a_T, a_C = ln_a_x.size() # batch size, sequence length, embedding dimensionality (n_embd)
a_q, a_k, a_v  = context_encoder.transformer.h[0].attn.c_attn(ln_a_x).split(n_embd, dim=2)

b_B, b_T, b_C = ln_b_x.size() # batch size, sequence length, embedding dimensionality (n_embd)
b_q, b_k, b_v  = context_encoder.transformer.h[0].attn.c_attn(ln_b_x).split(n_embd, dim=2)

assert torch.allclose(a_q, b_q[:, :len(test_a_inputs[0])])
assert torch.equal(a_q, b_q[:, :len(test_a_inputs[0])])
assert torch.allclose(a_k, b_k[:, :len(test_a_inputs[0])])
assert torch.equal(a_k, b_k[:, :len(test_a_inputs[0])])
assert torch.allclose(a_v, b_v[:, :len(test_a_inputs[0])])
assert torch.equal(a_v, b_v[:, :len(test_a_inputs[0])])

a_k = a_k.view(a_B, a_T, n_head, a_C // n_head).transpose(1, 2) # (B, nh, T, hs)
a_q = a_q.view(a_B, a_T, n_head, a_C // n_head).transpose(1, 2) # (B, nh, T, hs)
a_v = a_v.view(a_B, a_T, n_head, a_C // n_head).transpose(1, 2) # (B, nh, T, hs)

b_k = b_k.view(b_B, b_T, n_head, b_C // n_head).transpose(1, 2) # (B, nh, T, hs)
b_q = b_q.view(b_B, b_T, n_head, b_C // n_head).transpose(1, 2) # (B, nh, T, hs)
b_v = b_v.view(b_B, b_T, n_head, b_C // n_head).transpose(1, 2) # (B, nh, T, hs)

assert torch.allclose(a_q, b_q[:, :, :len(test_a_inputs[0])])
assert torch.equal(a_q, b_q[:, :, :len(test_a_inputs[0])])
assert torch.allclose(a_k, b_k[:, :, :len(test_a_inputs[0])])
assert torch.equal(a_k, b_k[:, :, :len(test_a_inputs[0])])
assert torch.allclose(a_v, b_v[:, :, :len(test_a_inputs[0])])
assert torch.equal(a_v, b_v[:, :, :len(test_a_inputs[0])])

a_q = context_encoder.rotary_embedding.rotate_queries_or_keys(a_q, indices = None)
a_k = context_encoder.rotary_embedding.rotate_queries_or_keys(a_k, indices = None)

b_q = context_encoder.rotary_embedding.rotate_queries_or_keys(b_q, indices = test_b_indices)
b_k = context_encoder.rotary_embedding.rotate_queries_or_keys(b_k, indices = test_b_indices)

assert torch.allclose(a_q, b_q[:, :, :len(test_a_inputs[0])])
assert torch.equal(a_q, b_q[:, :, :len(test_a_inputs[0])])
assert torch.allclose(a_k, b_k[:, :, :len(test_a_inputs[0])])
assert torch.equal(a_k, b_k[:, :, :len(test_a_inputs[0])])
assert torch.allclose(a_v, b_v[:, :, :len(test_a_inputs[0])])
assert torch.equal(a_v, b_v[:, :, :len(test_a_inputs[0])])

dropout = context_encoder.transformer.h[0].attn.dropout
training = context_encoder.transformer.h[0].attn.training

a_y = torch.nn.functional.scaled_dot_product_attention(a_q, a_k, a_v, attn_mask=None, dropout_p=dropout if training else 0)

b_y = torch.nn.functional.scaled_dot_product_attention(b_q, b_k, b_v, attn_mask=att_mask, dropout_p=dropout if training else 0)

assert torch.allclose(a_y, b_y[:, :, :len(test_a_inputs[0])])
assert torch.equal(a_y, b_y[:, :, :len(test_a_inputs[0])])

#%%
B, H, S, E = a_q.size()

#%%

B, H, S, E = a_q.size() 
attn_mask = torch.ones(B, S, dtype=torch.float).tril(diagonal=0) #if is_causal else attn_mask
attn_mask
#%%
attn_mask = attn_mask.masked_fill( attn_mask == 0, value = -float('inf'))#-float('inf'))
 #if attn_mask.dtype==torch.bool else attn_mask
attn_mask
#%%


#%%
def scaled_dot_product(Q, K, V, attn_mask = None, is_causal = False, dropout_p = 0.0, train = False):
    import math
    
    B, H, S, E = Q.size() 
    print(B, H, S, E)
    print(attn_mask.size() if attn_mask is not None else None)
    # attn_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0) if is_causal else attn_mask
    attn_mask = torch.ones(S, S, dtype=torch.bool) if attn_mask is None else attn_mask
    print(attn_mask)
    print(attn_mask.dtype)

    attn_mask = attn_mask.tril(diagonal=False) if is_causal else attn_mask
    print(attn_mask)
    print(attn_mask.dtype)

    attn_mask = attn_mask.float().masked_fill(attn_mask == False, -float('inf')) 
    print(attn_mask)
    print(attn_mask.dtype)
    
    attn_values = (Q @ K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
    print(attn_values)

    attn_values += attn_mask

    attn_weight = torch.softmax(attn_values, dim=-1)
    # attn_weight = torch.dropout(attn_weight, dropout_p, train = train)
    return attn_weight @ V


assert torch.equal(a_q, b_q[:, :, :len(test_a_inputs[0])])
assert torch.allclose(a_v, b_v[:, :, :len(test_a_inputs[0])])
assert torch.equal(a_k, b_k[:, :, :len(test_a_inputs[0])])

_a_y = scaled_dot_product(a_q, a_k, a_v, attn_mask=None, dropout_p=dropout if training else 0, is_causal=False)
_a_y

_b_y = scaled_dot_product(b_q, b_k, b_v, attn_mask=att_mask, dropout_p=dropout if training else 0, is_causal=False)
_b_y


new_att_mask = torch.zeros_like(test_b_inputs, dtype = torch.bool)
new_att_mask[:,:len(test_a_inputs[0])] = True
new_att_mask = new_att_mask[:, None, :] * new_att_mask[:, :, None]
new_att_mask = new_att_mask.unsqueeze(1)

new_att_mask = torch.block_diag(torch.ones(len(test_a_inputs[0]), len(test_a_inputs[0])), torch.ones((len(test_a_inputs[0]), len(test_a_inputs[0]))))
new_att_mask = torch.stack([new_att_mask for _ in range(len(test_b_inputs))])
new_att_mask = new_att_mask.to(torch.bool)
new_att_mask = new_att_mask.unsqueeze(1)


print("_b_y_2")
_b_y_2 = scaled_dot_product(b_q, b_k, b_v, attn_mask=new_att_mask, dropout_p=dropout if training else 0, is_causal=False)


assert torch.equal(_b_y[:, :, :len(test_a_inputs[0])], _b_y_2[:, :, :len(test_a_inputs[0])])

assert torch.allclose(_b_y_2[:, :, :len(test_a_inputs[0])], _a_y)


#%%
import math
#%%
a_attn_mask = None
a_B, a_H, a_S, a_E = a_q.size() 
# attn_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0) if is_causal else attn_mask
a_attn_mask = torch.ones(a_S, a_S, dtype=torch.bool) if a_attn_mask is None else a_attn_mask

a_attn_mask = a_attn_mask.float().masked_fill(a_attn_mask == False, -float('inf')) 

a_attn_values = (a_q @ a_k.transpose(-2, -1)) / math.sqrt(a_E)

print(a_q.dtype, a_k.dtype, a_v.dtype)

#%%
assert torch.allclose(a_q, b_q[:, :, :len(test_a_inputs[0])])
assert torch.equal(a_q, b_q[:, :, :len(test_a_inputs[0])])
assert torch.allclose(a_k, b_k[:, :, :len(test_a_inputs[0])])
assert torch.equal(a_k, b_k[:, :, :len(test_a_inputs[0])])
assert torch.allclose(a_v, b_v[:, :, :len(test_a_inputs[0])])
assert torch.equal(a_v, b_v[:, :, :len(test_a_inputs[0])])

b_y_2_attn_mask = new_att_mask
b_y_2_B, b_y_2_H, b_y_2_S, b_y_2_E = b_q.size()

b_y_2_attn_mask = torch.ones(b_y_2_S, b_y_2_S, dtype=torch.bool) if b_y_2_attn_mask is None else b_y_2_attn_mask

b_y_2_attn_mask = b_y_2_attn_mask.float().masked_fill(b_y_2_attn_mask == False, -float('inf'))

b_y_2_attn_values = (b_q @ b_k.transpose(-2, -1)) / math.sqrt(b_y_2_E)

#%%
print(b_q.dtype, b_k.dtype, b_v.dtype)

#%%
a_attn_values[:, :, :len(test_a_inputs[0]), :len(test_a_inputs[0])][0][0][0][0].item()
#%%
b_y_2_attn_values[:, :, :len(test_a_inputs[0]), :len(test_a_inputs[0])][0][0][0][0].item()
#%%
a_attn_values.shape
#%%%
assert torch.allclose(a_attn_values, b_y_2_attn_values[:, :, :len(test_a_inputs[0]), :len(test_a_inputs[0])])


#%%




#%%
attn_values += attn_mask

attn_weight = torch.softmax(attn_values, dim=-1)

#%%
new_att_mask.float().masked_fill(new_att_mask == False, -float('inf')) 

#%%
assert torch.allclose(_b_y[:, :, :len(test_a_inputs[0])], _a_y)


#%%
_b_y_2[:, :, :len(test_a_inputs[0])]

#%%
_a_y

#%%
assert torch.allclose(_b_y_2[:, :, :len(test_a_inputs[0])], _a_y)

#%%
assert torch.equal(_b_y_2[:, :, :len(test_a_inputs[0])], _b_y[:, :, :len(test_a_inputs[0])])

#%%
_b_y_2[:, :, :len(test_a_inputs[0])][0][0][0][0].item()
#%%
_b_y[:, :, :len(test_a_inputs[0])][0][0][0][0].item()

#%%
_a_y.shape
#%%
compare = _a_y
shape = compare.shape
for i in range(shape[0]):
    for j in range(shape[1]):
        for k in range(shape[2]):
            for l in range(shape[3]):
                if not compare[i][j][k][l].item() == _b_y_2[:, :, :len(test_a_inputs[0])][i][j][k][l].item():
                    print(compare[i][j][k][l].item(), _b_y_2[:, :, :len(test_a_inputs[0])][i][j][k][l].item())


for i in range(32):
    print(compare[0][0][0][i].item(), _b_y_2[:, :, :len(test_a_inputs[0])][0][0][0][i].item())

    if not compare[0][0][0][i].item() == _b_y_2[:, :, :len(test_a_inputs[0])][0][0][0][i].item():
        print(compare[0][0][0][i].item(), _b_y_2[:, :, :len(test_a_inputs[0])][0][0][0][i].item())


#%%
att_mask = torch.zeros_like(test_b_inputs, dtype = torch.bool)
att_mask[:,:len(test_a_inputs[0])] = True
att_mask = att_mask.float()

#%%
new_att_mask = torch.block_diag(torch.ones(len(test_a_inputs[0]), len(test_a_inputs[0])), torch.ones((len(test_a_inputs[0]), len(test_a_inputs[0]))))
new_att_mask

#%%
torch.block_diag(torch.ones(3, 3), torch.ones((3, 3)))

#%%

new_att_mask.shape



#%%
new_att_mask.unsqueeze(1).shape


#%%
new_att_mask = new_att_mask.bool()


#%%
test_att_mask = torch.ones(1, 3, dtype=torch.float)

test_att_mask[-1][-1] = 0
test_att_mask.shape
#%%
test_att_mask * test_att_mask.T


#%%

test_att_mask[:, test_att_mask.shape[-1] // 2  : ] = 0
test_att_mask


#%%
test_att_mask.shape[-1] // 2 

#%%
assert torch.allclose(b_y, _b_y, atol=1e-5)

#%%


#%%
a_x = a_x + context_encoder.transformer.h[0].attn(ln_a_x, context_encoder.rotary_embedding, None, None)
b_x = b_x + context_encoder.transformer.h[0].attn(ln_b_x, context_encoder.rotary_embedding, test_b_indices, att_mask)

assert torch.allclose(a_x, b_x[:, :len(test_a_inputs[0])])
assert torch.equal(a_x, b_x[:, :len(test_a_inputs[0])])


ln2_a_x = context_encoder.transformer.h[0].ln_2(a_x)
ln2_b_x = context_encoder.transformer.h[0].ln_2(b_x)

assert torch.allclose(ln2_a_x, ln2_b_x[:, :len(test_a_inputs[0])])


#%%

a_x = a_x + context_encoder.transformer.h[0].mlp(context_encoder.transformer.h[0].ln_2(a_x))
b_x = b_x + context_encoder.transformer.h[0].mlp(context_encoder.transformer.h[0].ln_2(b_x))

assert torch.allclose(a_x, b_x[:, :len(test_a_inputs[0])])


#%%

#%%
a_x = context_encoder.transformer.h[0](a_x, context_encoder.rotary_embedding, None, None)
b_x = context_encoder.transformer.h[0](b_x, context_encoder.rotary_embedding, test_b_indices, att_mask)

assert torch.allclose(a_x, b_x[:, :len(test_a_inputs[0])])


#%%
torch.exp(torch.tensor(-float('inf'))).item()