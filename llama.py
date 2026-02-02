import dataclasses
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file

import triton
import triton.language as tl

import ninetoothed as nt

@dataclasses.dataclass
class ModelConfig:
    head_dim: int

    hidden_size: int

    intermediate_size: int

    num_attention_heads: int

    num_hidden_layers: int

    num_key_value_heads: int

    rms_norm_eps: float

    rope_theta: float

    torch_dtype: str

    vocab_size: int


@triton.jit
def rms_norm_kernel(
    X, Y, W, 
    stride, n_cols, eps,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    X += row_idx * stride
    Y += row_idx * stride

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    
    # 将整行数据载入 SRAM，减少 HBM 访问次数
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    
    # 算子融合：在寄存器中直接计算
    var = tl.sum(x * x, axis=0) / n_cols
    rsqrt = tl.math.rsqrt(var + eps)
    
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    y = x * rsqrt * w
    
    tl.store(Y + cols, y, mask=mask)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(hidden_size))

        self.eps = eps

    def forward(self, input):
            # 记录原始形状以便最后还原
            orig_shape = input.shape
            # 将输入展平为 (num_rows, hidden_size) 以适配 Kernel
            x = input.view(-1, orig_shape[-1])
            M, N = x.shape
            y = torch.empty_like(x)
            BLOCK_SIZE = triton.next_power_of_2(N)
            
            # 启动 Triton Kernel
            rms_norm_kernel[(M,)](
                x, y, self.weight,
                x.stride(0), N, self.eps,
                BLOCK_SIZE=BLOCK_SIZE
            )
            return y.view(*orig_shape)


# 使用 jit 装饰器将 Python 函数编译为 GPU 算子
@nt.jit
def ninetoothed_mlp_op(gate, up, out):
    # 计算 SiLU: x * (1 / (1 + exp(-x)))
    # 直接使用 nt 提供的数学函数
    sig_g = 1.0 / (1.0 + nt.exp(-gate))
    out = (gate * sig_g) * up
    return out


@triton.jit
def mlp_fused_kernel(
    gate_ptr, up_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    # 每个 program 处理一个连续的块
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 加载 gate 和 up 的数据
    g = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
    u = tl.load(up_ptr + offsets, mask=mask).to(tl.float32)

    # 计算 SiLU: x * sigmoid(x)
    # tl.sigmoid 在 Triton 中已内置
    sig_g = tl.sigmoid(g)
    res = (g * sig_g) * u

    # 写回结果
    tl.store(out_ptr + offsets, res, mask=mask)


def triton_mlp_fusion(gate, up):
    n_elements = gate.numel()
    out = torch.empty_like(gate)
    
    # 定义并行网格，覆盖所有元素
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    mlp_fused_kernel[grid](
        gate, up, out,
        n_elements,
        BLOCK_SIZE=1024 # 可以根据 GPU 规格调整
    )
    return out



class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)

        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)

        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        self.silu = nn.SiLU()

    def forward(self, input):
        #return self.down_proj(self.silu(self.gate_proj(input)) * self.up_proj(input))
        
        # 1. 执行前两个线性投影
        gate_out = self.gate_proj(input)
        up_out = self.up_proj(input)
        
        # 2. 调用 Triton 融合算子 (替代原来的 self.silu(gate_out) * up_out)
        fused_out = triton_mlp_fusion(gate_out, up_out)
        
        # 3. 执行最后的下投影
        return self.down_proj(fused_out)
        '''
        gate_out = self.gate_proj(input)
        up_out = self.up_proj(input)
        
        # 准备输出张量
        fused_out = torch.empty_like(gate_out)
        
        # 直接调用，九齿会根据 Tensor 的 shape 自动映射线程
        ninetoothed_mlp_op(gate_out, up_out, fused_out)
        
        return self.down_proj(fused_out)
        '''

def apply_rotary_position_embedding(input, sin_table, cos_table):
    sin_table = sin_table[None, :, None, :]
    cos_table = cos_table[None, :, None, :]

    input_0 = input[..., : input.shape[-1] // 2]
    input_1 = input[..., input.shape[-1] // 2 :]
    input_0_rotated = input_0 * cos_table - input_1 * sin_table
    input_1_rotated = input_0 * sin_table + input_1 * cos_table

    return torch.cat((input_0_rotated, input_1_rotated), dim=-1)


def apply_scaled_dot_product_attention(query, key, value):
    _, num_heads_q, seq_len_q, emb_dim = query.shape
    _, num_heads_k, seq_len_k, _ = key.shape
    _, num_heads_v, _, _ = value.shape

    key = key.repeat_interleave(num_heads_q // num_heads_k, 1)
    value = value.repeat_interleave(num_heads_q // num_heads_v, 1)

    scale = 1 / math.sqrt(emb_dim)
    attn_mask = torch.tril(
        torch.full((seq_len_q, seq_len_k), True, device=query.device)
    )

    attn_output = torch.matmul(query, key.permute(0, 1, 3, 2)) * scale
    attn_output = torch.where(attn_mask, attn_output, float("-inf"))
    attn_output = torch.softmax(attn_output, dim=-1)
    attn_output = torch.matmul(attn_output, value)

    return attn_output


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.head_dim = config.head_dim

        self.hidden_size = config.hidden_size

        self.num_attention_heads = config.num_attention_heads

        self.num_key_value_heads = config.num_key_value_heads

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_attention_heads * self.head_dim, bias=False
        )

        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )

        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )

        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim, self.hidden_size, bias=False
        )

    def forward(self, hidden_states, sin_table, cos_table):
        batch_size, seq_len = hidden_states.shape[:2]
        hidden_shape = (batch_size, seq_len, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape).permute(0, 2, 1, 3)

        query_states = apply_rotary_position_embedding(
            query_states, sin_table, cos_table
        ).permute(0, 2, 1, 3)
        key_states = apply_rotary_position_embedding(
            key_states, sin_table, cos_table
        ).permute(0, 2, 1, 3)

        attn_output = apply_scaled_dot_product_attention(
            query_states, key_states, value_states
        )

        return self.o_proj(
            attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        )


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        self.self_attn = Attention(config)

        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        self.mlp = MLP(config.hidden_size, config.intermediate_size)

    def forward(self, hidden_states, sin_table, cos_table):
        hidden_states += self.self_attn(
            self.input_layernorm(hidden_states), sin_table, cos_table
        )

        hidden_states += self.mlp(self.post_attention_layernorm(hidden_states))

        return hidden_states


def generate_sin_and_cos_tables(seq_len, emb_dim, base, dtype, device):
    theta = base ** (
        -2 * (torch.arange(emb_dim // 2, dtype=dtype, device=device) / emb_dim)
    )

    positions = torch.arange(seq_len, dtype=dtype, device=device).unsqueeze(1)
    sin_table = torch.sin(positions * theta)
    cos_table = torch.cos(positions * theta)

    return sin_table, cos_table


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.head_dim = config.head_dim

        self.hidden_size = config.hidden_size

        self.num_hidden_layers = config.num_hidden_layers

        self.rms_norm_eps = config.rms_norm_eps

        self.rope_theta = config.rope_theta

        self.torch_dtype = config.torch_dtype

        self.vocab_size = config.vocab_size

        self.embed_tokens = torch.nn.Embedding(self.vocab_size, self.hidden_size)

        self.layers = nn.ModuleList(
            DecoderLayer(config) for _ in range(self.num_hidden_layers)
        )

        self.norm = RMSNorm(self.hidden_size, self.rms_norm_eps)

    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)

        seq_len = hidden_states.shape[1]

        sin_table, cos_table = generate_sin_and_cos_tables(
            seq_len,
            self.head_dim,
            base=self.rope_theta,
            dtype=getattr(torch, self.torch_dtype),
            device=input_ids.device,
        )

        for i in range(self.num_hidden_layers):
            hidden_states = self.layers[i](hidden_states, sin_table, cos_table)

        return self.norm(hidden_states)


class ModelForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.model = Model(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def generate(self, input_ids, max_new_tokens=20):
        for _ in range(max_new_tokens):
            hidden_states = self.model(input_ids)

            logits = self.lm_head(hidden_states[:, -1, :])

            next = torch.argmax(logits, dim=-1).unsqueeze(-1)

            input_ids = torch.cat((input_ids, next), dim=-1)

        return input_ids

    @staticmethod
    def from_pretrained(model_path):
        model_path = Path(model_path)

        with open(model_path / "config.json") as f:
            config = json.load(f)

        if "head_dim" not in config:
            config["head_dim"] = config["hidden_size"] // config["num_attention_heads"]

        config = ModelConfig(
            **{
                key: value
                for key, value in config.items()
                if key in ModelConfig.__annotations__
            }
        )

        model = ModelForCausalLM(config).to(getattr(torch, config.torch_dtype))

        state_dict = load_file(model_path / "model.safetensors")

        if "lm_head.weight" not in state_dict:
            state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]

        model.load_state_dict(state_dict)

        return model
