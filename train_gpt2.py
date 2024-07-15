from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import time

# ------------------------------------------------------------------------------

@dataclass
class GPTConfig:
    """Describes a GPT model structure"""
    block_size: int = 1024  # the size of our context window (max sequences length)
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes + 1 special end of document token
    n_layer:    int = 12    # number of layers
    n_head:     int = 12    # number of heads
    n_embd:     int = 768   # the size of the embedding vector

class CausalSelfAttention(nn.Module):
    """Attention mechanism used in the transformer block"""
    def __init__(self,
                 n_embd: int,
                 n_head: int,
                 block_size: int):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3*n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1 # special init scaling for the output of this module
        # regularization
        self.n_head = n_head
        self.n_embd = n_embd
        # not really a "bias", more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))
        
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence legth, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M): nh=12, hs=64, so nh*hs = C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        nh = self.n_head
        hs = C // self.n_head
        k = k.view(B, T, nh, hs).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, nh, hs).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, nh, hs).transpose(1, 2) # (B, nh, T, hs)
        
        # attention (materializes the large (T,T) matrix for all the queries and keys)    
        if torch.cuda.is_available():
            # Use flash attention.
            # This is a fused kernel that takes advantage of the 2018 Nvidia algorithm for
            # calculating the softmax in a streaming fashion.
            # So the whole attention mechanism is done without having to materialize the attention
            # matrix in the HBM.
            # This algorithm uses much higher FLOPs, but completely avoids costly memory accesses.
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # the slower, more explicit way
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # k.size(-1) should be hs
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    """Multi-layer perceptron, or feed-forward network"""
    def __init__(self, n_embd: int, mult=4):
        super().__init__()
        self.c_fc   = nn.Linear(n_embd, n_embd*mult)
        self.gelu   = nn.GELU(approximate='tanh')  # GPT-2 used approximate, otherwise the exact version is preferred now and performs well
        self.c_proj = nn.Linear(n_embd*mult, n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1 # special init scaling for the output of this module
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    """The transformer block."""

    # TODO: pass which specific configs we need, rather than whole config object
    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)
    
    def forward(self, x):
        # notice the "x +" residual connections
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp( self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config: GPTConfig):  # Added GPTConfig type annotation
        super().__init__()
        self.config = config

        # Build the structure of the model
        # We'll initialize the weights later
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([
                Block(config.n_embd, config.n_head, config.block_size)
                for _ in range(config.n_layer)
            ]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        # translate the embedding back to a token
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight sharing scheme
        # We reuse the Linear layer's weights, as the Embedding layer's weights are initialized
        # in a less optimal way for this use case.
        # (If we set them up in the reverse way, we get a loss of 400+ instead of 10.8.)
        self.transformer.wte.weight = self.lm_head.weight

        # init params with special logic
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights using special logic from GPT2 paper"""
        if isinstance(module, (nn.Linear)):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                # the layers stack on one another, we want to keep the std from growing out of hand
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        # idx is of shape (B, T) of integers
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence length {T} > block size {self.config.block_size}"
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        loss = None
        if targets != None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),  # (B * T, vocab_size)
                targets.view(-1) # (B * T)
            )
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type: str):
        """Load a pre-trained model from the Hugging Face model hub"""
        assert model_type in [
            "gpt2",
            "gpt2-medium",
            "gpt2-large",
            "gpt2-xl"
        ]
        from transformers import GPT2LMHeadModel
        print(f"Loading {model_type} pretrained model weights...")
        # n_layer, n_head, and n_embd are determined from model type
        config_args = {
            'gpt2':        dict(n_layer=12, n_head=12, n_embd=768),   #  124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  #  350M params
            'gpt2-large':  dict(n_layer=36, n_head=20, n_embd=1280),  #  774M params
            'gpt2-xl':     dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        # set options that are the same for all GPT2 model checkpoints
        config_args['vocab_size'] = 50304 # increased from 50257 to be a nice multiple of 128
        config_args['block_size'] = 1024
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask/buffer
        
        # # print out some info about the state dict
        # print(f"State dict has {len(sd)} entries, {len(sd_keys)} after filtering.")
        # for k in sd_keys:
        #     v = sd[k]
        #     print(f"  {k}\t{v.shape} = {v.view(-1).size().numel():,}")
        # sum up the sizes
        total_params = sum([sd[k].view(-1).size().numel() for k in sd_keys])
        print(f"Total parameters: {total_params:,}")

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_hf = model_hf.state_dict()
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these buffers
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        # The openai checkpoints use a 'Conv1D" module, but we only want to use a vanilla Linear layer.
        # This means that we have to transpose these weights when we import them
        transposed = [
            'attn.c_attn.weight',
            'attn.c_proj.weight',
            'mlp.c_fc.weight',
            'mlp.c_proj.weight',
        ]
        with torch.no_grad():
            for k in sd_keys_hf:
                source = sd_hf[k]
                if any(k.endswith(w) for w in transposed):
                    # for the Conv1D weights we need to transpose
                    source = source.t()
                # copy over the parameters
                assert source.shape == sd[k].shape, f"copying {k}: {source.shape} != {sd[k].shape}"
                sd[k].copy_(source)

        return model

# ------------------------------------------------------------------------------
import tiktoken

class DataLoaderLite:
    def __init__(self, filename, B, T):
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        with open(filename, 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        print(f"Loaded {len(self.tokens):,} tokens.")
        print(f"1 epoch = {len(self.tokens) // (B * T):,} batches")

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        x = buf[:-1].view(B, T) # inputs
        y = buf[1:].view(B, T)  # targets
        # advance the position of the tensor
        self.current_position += B*T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B*T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y

# ------------------------------------------------------------------------------
import sys

# attempt to autodetect the device
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
print(f"Using device: {device}")

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# get a data batch
train_loader = DataLoaderLite('tiny_shakespeare.txt', B=4, T=256)

if device == 'cuda':
    # A100's can use tensor float 32 for matmuls, which drops bits of precision from the mantissa.
    # It's theoretically 8x faster, but we might observe only 3x faster.
    torch.set_float32_matmul_precision('high')

# x, y = train_loader.next_batch()
# print(x.shape, y.shape)
# # display x and y as decoded text
# enc = tiktoken.get_encoding("gpt2")
# for i in range(x.size(0)):
#     print(f"\nBatch {i+1}:")
#     print("X:", [enc.decode([t]) for t in x[i].tolist()])
#     print("Y:", [enc.decode([t]) for t in y[i].tolist()])

# construct model
model = GPT(GPTConfig())
model.to(device)
# don't compile the model on CPU, catastrophic performance degradation!
if device == 'cuda':
    print("Compiling model")
    model = torch.compile(model)

steps = 5
print(f"Performing {steps} optimization steps")
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(5):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)

    # remember to zero the gradients before running the backward pass
    optimizer.zero_grad()

    if device == 'cuda':
        # enable autocast to bfloat16 only during the forward pass (model + loss)
        # this hopefully gives some more improved performance (at the cost of some precision).
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
    else:
        logits, loss = model(x, y)

    loss.backward()
    optimizer.step()
    if device == 'cuda':
        torch.cuda.synchronize() 
    dt = time.time() - t0
    tokens_per_sec = (train_loader.B * train_loader.T) / dt
    print(f"step #{i+1}), loss: {loss.item():.6f}, dt: {dt*1000:.2f}ms, tokens/sec: {tokens_per_sec:.2f}")

with torch.no_grad():
    _, loss = model(x, y)
    print(f"Final loss: {loss.item()}")


sys.exit(0)



# load the model
model = GPT.from_pretrained("gpt2")
# model = GPT(GPTConfig()) # randomly initialized model
model.eval() # put in eval mode for generating; not sure if it actually matters for this gpt2 model
model.to(device)

# generate some text
prompt = "Hello, I'm a language model,"
num_return_sequences = 3
max_length = 30
print(f"Generating {num_return_sequences} completions for prompt: {prompt}")

# prefix tokens
tokens = enc.encode(prompt)
tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
x = tokens.to(device)

# generate! right now x is (B, T) where B=5, T=8
# set the seed to 42
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x) # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        ix = torch.multinomial(topk_probs, num_samples=1) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(f"#{i+1}) {decoded}")