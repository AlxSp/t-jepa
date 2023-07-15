#%%
import torch
from rotary_embedding_torch import RotaryEmbedding

# instantiate the positional embedding in your transformer and pass to all your attention layers

rotary_emb = RotaryEmbedding(dim = 32)

# mock queries and keys - dimensions should end with (seq_len, feature dimension), and any number of preceding dimensions (batch, heads, etc)
batch_size = 2

q = torch.randn(batch_size, 8, 1024, 64) # queries - (batch, heads, seq len, dimension of head)
k = torch.randn(batch_size, 8, 1024, 64) # keys

q = torch.arange(batch_size * 8 * 1024 * 64).reshape(batch_size, 8, 1024, 64)

# apply the rotations to your queries and keys after the heads have been split out, but prior to the dot product and subsequent softmax (attention)
#%%
rq = rotary_emb.rotate_queries_or_keys(q)
k = rotary_emb.rotate_queries_or_keys(k, indices=torch.arange(1024 * 2).view(2, 1024))

print(rq)

#%%
print(q.shape)

#%%
i_rq = rotary_emb.rotate_queries_or_keys(q, indices=torch.arange(1024).unsqueeze(0))

assert torch.allclose(rq, i_rq)

#%%

idx = torch.randperm(1024)
assert torch.allclose(rq, rotary_emb.rotate_queries_or_keys(q, indices=torch.stack([torch.arange(1024) for b in range(batch_size)]) )), 'rotary embeddings should be the same when indices are the same'

#%%
batch_ranges = []
for b in range(batch_size):
    batch_ranges.append(torch.arange(1024) + b * 1024)
batch_ranges = torch.stack(batch_ranges)

individually_rotated = [rotary_emb.rotate_queries_or_keys(q[b].unsqueeze(0), indices=batch_ranges[b].unsqueeze(0)) for b in range(batch_size)]

batched_rotated = rotary_emb.rotate_queries_or_keys(q, indices=batch_ranges)

assert torch.allclose(torch.cat(individually_rotated), batched_rotated), 'rotary embeddings should be the same when indices are the same'

#%%
batch_ranges[b].unsqueeze(0).shape
batch_ranges.shape
#%%
q.shape
#%%
q[b].shape

#%%
q[b].unsqueeze(0).shape

#%%
torch.stack(individually_rotated).shape

#%%
n_inputs = 1024
n_total = 1024 * 8
index_ranges = torch.arange(1024)
mod_index_ranges = index_ranges[idx]
i_rq_2 = rotary_emb.rotate_queries_or_keys(q[:,:,mod_index_ranges], indices=mod_index_ranges.unsqueeze(0))

assert torch.allclose(rq[:,:,mod_index_ranges], i_rq_2), 'rotary embeddings should be the same when indices are the same'


unshuf_order = torch.zeros_like(mod_index_ranges)
unshuf_order[mod_index_ranges] = torch.arange(1024)

unshuffled_data = i_rq_2[:,:,unshuf_order] # Unshuffle the shuffled data
assert torch.allclose(rq, unshuffled_data), 'rotary embeddings should be the same when indices are the same'

i_rq_off = rotary_emb.rotate_queries_or_keys(q[:,:,mod_index_ranges], indices=mod_index_ranges.unsqueeze(0) + 1)

assert not torch.allclose(rq[:,:,mod_index_ranges], i_rq_off), 'rotary embeddings should be different when indices are different'

#%%

assert not torch.allclose(rq, i_rq_2), 'rotary embeddings should be different when indices are different'


#%%
i_rq_2[:,:,783]

#%%
rq[:,:,0]

#%%
