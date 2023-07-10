#%%
import torch
from rotary_embedding_torch import RotaryEmbedding

# instantiate the positional embedding in your transformer and pass to all your attention layers

rotary_emb = RotaryEmbedding(dim = 32)

# mock queries and keys - dimensions should end with (seq_len, feature dimension), and any number of preceding dimensions (batch, heads, etc)

q = torch.randn(1, 8, 1024, 64) # queries - (batch, heads, seq len, dimension of head)
k = torch.randn(1, 8, 1024, 64) # keys

q = torch.arange(8 * 1024 * 64).reshape(1, 8, 1024, 64)

# apply the rotations to your queries and keys after the heads have been split out, but prior to the dot product and subsequent softmax (attention)
#%%
rq = rotary_emb.rotate_queries_or_keys(q)
k = rotary_emb.rotate_queries_or_keys(k)

print(rq)

#%%
i_rq = rotary_emb.rotate_queries_or_keys(q, indices=torch.arange(1024))

print(rq)

assert torch.allclose(rq, i_rq)

#%%
n_inputs = 1024
n_total = 1024 * 8
index_ranges = torch.arange(1024)
idx = torch.randperm(1024)
mod_index_ranges = index_ranges[idx]
i_rq_2 = rotary_emb.rotate_queries_or_keys(q[:,:,mod_index_ranges], indices=mod_index_ranges)

assert torch.allclose(rq[:,:,mod_index_ranges], i_rq_2), 'rotary embeddings should be the same when indices are the same'


unshuf_order = torch.zeros_like(mod_index_ranges)
unshuf_order[mod_index_ranges] = torch.arange(1024)

unshuffled_data = i_rq_2[:,:,unshuf_order] # Unshuffle the shuffled data
assert torch.allclose(rq, unshuffled_data), 'rotary embeddings should be the same when indices are the same'

i_rq_off = rotary_emb.rotate_queries_or_keys(q[:,:,mod_index_ranges], indices=mod_index_ranges + 1)

assert not torch.allclose(rq[:,:,mod_index_ranges], i_rq_off), 'rotary embeddings should be different when indices are different'

#%%

assert not torch.allclose(rq, i_rq_2), 'rotary embeddings should be different when indices are different'


#%%
i_rq_2[:,:,783]

#%%
rq[:,:,0]

#%%
