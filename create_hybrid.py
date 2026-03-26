import re

with open("train_gpt.py", "r") as f:
    code = f.read()

# Add DeltaNet imports / helper functions before CausalSelfAttention
delta_net_code = """
def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm

def torch_chunk_gated_delta_rule(
    query, key, value, g, beta, chunk_size=128, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=False
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    if pad_size > 0:
        query = F.pad(query, (0, 0, 0, pad_size))
        key = F.pad(key, (0, 0, 0, pad_size))
        value = F.pad(value, (0, 0, 0, pad_size))
        beta = F.pad(beta, (0, pad_size))
        g = F.pad(g, (0, pad_size))
        
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
    
    core_attn_out = torch.zeros_like(value)
    mask_2 = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn_out = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask_2, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn_out @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state

class GatedRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x, gate):
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = self.weight * x.to(orig_dtype)
        x = x * F.silu(gate.to(torch.float32)).to(orig_dtype)
        return x

class GatedDeltaNet(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        conv_kernel_size: int = 4
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads 
        self.num_kv_heads = max(1, num_kv_heads) 
        self.head_v_dim = dim // num_heads
        self.head_k_dim = dim // self.num_kv_heads
        self.key_dim = self.head_k_dim * self.num_kv_heads
        self.value_dim = self.head_v_dim * self.num_heads
        
        self.conv_kernel_size = conv_kernel_size

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )
        
        self.dt_bias = nn.Parameter(torch.ones(self.num_heads))
        self.A_log = nn.Parameter(torch.log(torch.empty(self.num_heads).uniform_(0, 16)))
        
        self.norm = GatedRMSNorm(self.head_v_dim)
        
        self.out_proj = CastedLinear(self.value_dim, self.dim, bias=False)
        self.out_proj._zero_init = True
        
        self.in_proj_qkv = CastedLinear(self.dim, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = CastedLinear(self.dim, self.value_dim, bias=False)
        self.in_proj_b = CastedLinear(self.dim, self.num_heads, bias=False)
        self.in_proj_a = CastedLinear(self.dim, self.num_heads, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        mixed_qkv = self.in_proj_qkv(hidden_states)
        mixed_qkv = mixed_qkv.transpose(1, 2)
        
        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)
        
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])
        mixed_qkv = mixed_qkv.transpose(1, 2)
        
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )
        
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)
        
        beta = b.sigmoid()
        # Ensure log values don't overflow fp16. 
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        
        if self.num_heads // self.num_kv_heads > 1:
            query = query.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)
            key = key.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)
            
        core_attn_out, _ = torch_chunk_gated_delta_rule(
            query, key, value, g=g, beta=beta,
            chunk_size=128, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True
        )
        
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        
        return self.out_proj(core_attn_out)

class CausalSelfAttention"""

code = code.replace("class CausalSelfAttention", delta_net_code)

# Update Block __init__
block_init_old = """    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)"""

block_init_new = """    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        layer_idx: int,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.layer_idx = layer_idx
        # Hybrid structure: 2 delta net layers, 1 full attention layer
        self.is_linear_attention = (layer_idx % 3 != 2)
        if self.is_linear_attention:
            self.attn = GatedDeltaNet(dim, num_heads, num_kv_heads)
        else:
            self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)"""

code = code.replace(block_init_old, block_init_new)

# Update GPT blocks instantiation
gpt_blocks_old = """        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                )
                for i in range(num_layers)
            ]
        )"""

gpt_blocks_new = """        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                    layer_idx=i,
                )
                for i in range(num_layers)
            ]
        )"""

code = code.replace(gpt_blocks_old, gpt_blocks_new)

# Add control tensor names
control_tensors_old = '"attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights"'
control_tensors_new = '"attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,dt_bias,A_log"'

code = code.replace(control_tensors_old, control_tensors_new)

# Allow compiling new fn
compile_fn_add = "zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)"
compile_fn_new = """zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)
    global torch_chunk_gated_delta_rule
    torch_chunk_gated_delta_rule = torch.compile(torch_chunk_gated_delta_rule, dynamic=False, fullgraph=True)"""

code = code.replace(compile_fn_add, compile_fn_new)

with open("train_hybrid.py", "w") as f:
    f.write(code)
