import dataclasses
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file

import triton
import triton.language as tl

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

# 加速点一
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

# 加速点二
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

# 加速点三 
@triton.jit
def softmax_kernel(
    output_ptr, input_ptr,
    stride_b, stride_h, stride_m, stride_n,  # stride_m 是行跨度, stride_n 是列跨度
    n_rows, n_cols,
    scale,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    
    # 1. 计算当前行的起始指针
    # 修正点 A: 行索引 row_idx 应该乘以 行跨度 stride_m
    row_start_ptr = input_ptr + row_idx * stride_m
    
    # 2. 计算列偏移
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # 3. 生成输入数据的指针块 (Block of Pointers)
    # 指针 = 行首 + 列偏移 * 列跨度
    input_ptrs = row_start_ptr + col_offsets * stride_n
    
    # 加载数据
    row = tl.load(input_ptrs, mask=mask, other=-float('inf')).to(tl.float32)
    row = row * scale
    
    # 4. Softmax 计算 (Online Softmax)
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    output = numerator / denominator
    
    # 5. 写回数据
    # 修正点 B: 输出指针也必须加上 col_offsets * stride_n，变成向量指针
    output_ptrs = output_ptr + row_idx * stride_m + col_offsets * stride_n
    tl.store(output_ptrs, output, mask=mask)

def triton_softmax(x, scale=1.0):
    # x shape: (..., n_cols)
    n_rows = x.numel() // x.shape[-1]
    n_cols = x.shape[-1]
    
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    # 将输入视为 (n_rows, n_cols) 的二维矩阵
    # stride(-2) 是行跨度，stride(-1) 是列跨度
    grid = (n_rows, )
    
    softmax_kernel[grid](
        out, x,
        0, 0, x.stride(-2), x.stride(-1), # 传入 stride_m, stride_n
        n_rows, n_cols,
        scale,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

# 加速点四
# ==========================================
# Triton RoPE (旋转位置编码) 融合算子
# ==========================================

@triton.jit
def rope_kernel(
    q_ptr,           # Query/Key 输入指针 (Batch, Seq, Head, Dim)
    cos_ptr,         # Cos 表指针 (Seq, Dim)
    sin_ptr,         # Sin 表指针 (Seq, Dim)
    out_ptr,         # 输出指针
    q_row_stride,    # Q 的行跨度 (通常是 Head * Dim)
    q_head_stride,   # Q 的头跨度 (通常是 Dim)
    cos_row_stride,  # Cos 的行跨度
    sin_row_stride,  # Sin 的行跨度
    n_seq,           # 序列长度
    n_head,          # 头数
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr, # 也就是 Dim
    HALF_DIM: tl.constexpr  # Dim // 2
):
    # 并行策略：每个 Program 处理一个 (Seq, Head) 组合
    # Grid: (n_seq, n_head * batch_size)
    # pid_seq = tl.program_id(0)
    # pid_bh  = tl.program_id(1) (Batch * Head)
    
    pid_seq = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    # 1. 计算 Q/K 的内存偏移
    # 这里的 q_ptr 已经包含了 batch 的偏移，所以在 Kernel 里只需要处理 seq 和 head
    # 实际上为了简单，我们可以在外层就把 q_ptr 指向当前 Batch 的开始，或者把 Batch 维度合并到 pid_bh 中
    # 假设输入是展平的 (Total_Seq, Head, Dim) 或者 stride 能够处理
    
    # 偏移量计算：
    # Batch_Head_Index * Head_Stride + Seq_Index * Row_Stride (注意 Llama 的 layout)
    # 原始 shape: (Batch, Seq, Head, Dim)
    # 但 llama.py 里是 (Batch, Seq, Head, Dim)
    # 对应的 stride: 
    #   stride(0) = Seq * Head * Dim (Batch)
    #   stride(1) = Head * Dim       (Seq)  <- q_row_stride
    #   stride(2) = Dim              (Head) <- q_head_stride
    #   stride(3) = 1
    
    # 我们用 grid (n_seq, batch * n_head)
    # batch_id = pid_bh // n_head
    # head_id  = pid_bh % n_head
    
    # 对应数据的指针位置：
    # ptr = base + batch_id * stride_batch + seq_id * stride_seq + head_id * stride_head
    # 为简化计算，我们在 Python 端可以直接传入 stride
    
    q_offset = pid_bh * q_head_stride + pid_seq * q_row_stride
    cos_offset = pid_seq * cos_row_stride
    
    # 2. 加载 Cos 和 Sin
    # RoPE 的特点：前半部分和后半部分共享 Cos/Sin，或者说对应位置
    # Llama 采用 rotate_half: x[i] 与 x[i + half] 配对
    # Cos/Sin 表通常也是 (Seq, Dim)
    
    # 加载前半部分索引 [0, 1, ... HALF_DIM-1]
    offs_half = tl.arange(0, HALF_DIM)
    
    # 加载 Cos/Sin (前一半)
    # 注意：generate_sin_and_cos_tables 生成的是完整 Dim 长度，但前后半部分值是一样的(对于 theta)
    # 不过代码里是 cat 起来的，我们直接读对应位置即可
    cos0 = tl.load(cos_ptr + cos_offset + offs_half)
    sin0 = tl.load(sin_ptr + cos_offset + offs_half)
    
    # 3. 加载 Q/K 数据
    # 加载前半部分 x_0
    q0_ptr = q_ptr + q_offset + offs_half
    q0 = tl.load(q0_ptr).to(tl.float32)
    
    # 加载后半部分 x_1
    q1_ptr = q_ptr + q_offset + offs_half + HALF_DIM
    q1 = tl.load(q1_ptr).to(tl.float32)
    
    # 4. 执行旋转 (Rotate Half)
    # x_0_new = x_0 * cos - x_1 * sin
    # x_1_new = x_0 * sin + x_1 * cos
    
    q0_out = q0 * cos0 - q1 * sin0
    q1_out = q0 * sin0 + q1 * cos0
    
    # 5. 写回 (In-place 修改，节省显存)
    tl.store(out_ptr + q_offset + offs_half, q0_out)
    tl.store(out_ptr + q_offset + offs_half + HALF_DIM, q1_out)

def triton_apply_rope(x, cos, sin):
    # x shape: (Batch, Seq, Head, Dim)
    # cos/sin shape: (Seq, Dim)
    
    batch, seq_len, n_head, head_dim = x.shape
    half_dim = head_dim // 2
    
    # 创建输出张量 (或者直接 clone 后原地修改)
    out = torch.empty_like(x)
    
    # 确保连续，否则 stride 可能会乱
    # x = x.contiguous() 
    # cos = cos.contiguous()
    # sin = sin.contiguous()
    
    # 启动 Kernel
    # Grid 维度: (Seq_Len, Batch * Num_Heads)
    grid = (seq_len, batch * n_head)
    
    # stride 计算
    # x.stride(1) 是 seq 维度的 stride
    # x.stride(2) 是 head 维度的 stride
    # 注意：如果 Batch * Head 被合并到了 grid[1]，我们需要手动计算 batch 带来的 stride 偏移
    # 为了简化，我们假设 x 是 (Batch, Seq, Head, Dim) 布局
    # 我们在 Kernel 里把 stride 传得更细一点
    
    # 这里有个小技巧：我们可以把 Batch 和 Head 视为一个大维度，
    # 但前提是 Batch 和 Head 在内存上不一定是连续的（中间隔了 Seq）
    # Llama 的 Tensor 布局通常是 (Batch, Seq, Head, Dim)
    # 所以 Batch 维度 stride 最大，Seq 次之，Head 再次之
    
    # 为了适配 Kernel 的 (pid_bh * head_stride + pid_seq * row_stride) 公式：
    # pid_bh 包含了 batch 和 head。
    # 真实的 offset = batch_idx * stride_b + head_idx * stride_h + seq_idx * stride_s
    # = (pid_bh // n_head) * stride_b + (pid_bh % n_head) * stride_h + ...
    # 这在 Kernel 里算太慢。
    
    # 简单方案：直接用 view 展平 Batch 和 Head？
    # x.transpose(1, 2) -> (Batch, Head, Seq, Dim).reshape(-1, Seq, Dim)
    # 这样内存就需要拷贝了，得不偿失。
    
    # 最佳方案：直接在 Kernel 里传 stride
    # 但我们不想在 Kernel 做除法。
    # 妥协方案：只针对 seq 维度并行，内部循环 batch 和 head？不，并行度不够。
    
    # 让我们修正 Kernel 逻辑：
    # 传入 stride_batch, stride_seq, stride_head
    
    rope_kernel_advanced[(seq_len, batch * n_head)](
        x, cos, sin, out,
        x.stride(0), x.stride(1), x.stride(2), # stride_batch, stride_seq, stride_head
        cos.stride(0), sin.stride(0),
        BLOCK_SIZE=triton.next_power_of_2(half_dim),
        HEAD_DIM=head_dim,
        HALF_DIM=half_dim,
        N_HEAD=n_head # 传进去做取模运算
    )
    return out

# 修正后的 Kernel (支持非连续维度的 Batch/Head)
@triton.jit
def rope_kernel_advanced(
    q_ptr, cos_ptr, sin_ptr, out_ptr,
    stride_batch, stride_seq, stride_head,
    cos_stride_seq, sin_stride_seq,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF_DIM: tl.constexpr,
    N_HEAD: tl.constexpr
):
    pid_seq = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    # 反解 Batch 和 Head ID
    batch_id = pid_bh // N_HEAD
    head_id = pid_bh % N_HEAD
    
    # 计算 Q 的偏移
    q_offset = batch_id * stride_batch + pid_seq * stride_seq + head_id * stride_head
    
    # 计算 Cos/Sin 的偏移 (只跟 Seq 有关)
    cos_offset = pid_seq * cos_stride_seq
    
    offs = tl.arange(0, HALF_DIM)
    
    # 加载
    cos = tl.load(cos_ptr + cos_offset + offs)
    sin = tl.load(sin_ptr + cos_offset + offs)
    
    q0 = tl.load(q_ptr + q_offset + offs).to(tl.float32)
    q1 = tl.load(q_ptr + q_offset + offs + HALF_DIM).to(tl.float32)
    
    # 计算
    out0 = q0 * cos - q1 * sin
    out1 = q0 * sin + q1 * cos
    
    # 存储
    tl.store(out_ptr + q_offset + offs, out0)
    tl.store(out_ptr + q_offset + offs + HALF_DIM, out1)
    

def apply_rotary_position_embedding(input, sin_table, cos_table):
    '''
    sin_table = sin_table[None, :, None, :]
    cos_table = cos_table[None, :, None, :]

    input_0 = input[..., : input.shape[-1] // 2]
    input_1 = input[..., input.shape[-1] // 2 :]
    input_0_rotated = input_0 * cos_table - input_1 * sin_table
    input_1_rotated = input_0 * sin_table + input_1 * cos_table

    return torch.cat((input_0_rotated, input_1_rotated), dim=-1)
    '''
    return triton_apply_rope(input, cos_table, sin_table)

def apply_scaled_dot_product_attention(query, key, value):
    _, num_heads_q, seq_len_q, emb_dim = query.shape
    _, num_heads_k, seq_len_k, _ = key.shape
    _, num_heads_v, _, _ = value.shape

    key = key.repeat_interleave(num_heads_q // num_heads_k, 1)
    value = value.repeat_interleave(num_heads_q // num_heads_v, 1)

    scale = 1 / math.sqrt(emb_dim)

    # 1. Q @ K.T
    attn_weights = torch.matmul(query, key.permute(0, 1, 3, 2))
    
    # 2. 应用 Mask (关键修正！)
    # 只有在 Prefill 阶段 (seq_len_q > 1) 才需要因果掩码
    # 在 Decode 阶段 (seq_len_q == 1)，Query 应该能看到之前所有的 Key
    if seq_len_q > 1:
        attn_mask = torch.tril(
            torch.full((seq_len_q, seq_len_k), True, device=query.device)
        )
        attn_weights = torch.where(attn_mask, attn_weights, float("-inf"))
    
    # 3. Triton Softmax
    attn_weights = triton_softmax(attn_weights, scale=scale)
    
    # 4. Score @ V
    attn_output = torch.matmul(attn_weights, value)
    
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

    def forward(self, hidden_states, sin_table, cos_table, past_key_value=None):
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

        # KV Cache 拼接逻辑
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat((past_key, key_states), dim=2)
            value_states = torch.cat((past_value, value_states), dim=2)
        
        current_key_value = (key_states, value_states)

        attn_output = apply_scaled_dot_product_attention(
            query_states, key_states, value_states
        )

        return self.o_proj(
            attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        ), current_key_value


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        self.self_attn = Attention(config)

        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        self.mlp = MLP(config.hidden_size, config.intermediate_size)

    '''
    def forward(self, hidden_states, sin_table, cos_table):
        hidden_states += self.self_attn(
            self.input_layernorm(hidden_states), sin_table, cos_table
        )

        hidden_states += self.mlp(self.post_attention_layernorm(hidden_states))

        return hidden_states
    '''
    def forward(self, hidden_states, sin_table, cos_table, past_key_value=None):
        attn_output, current_key_value = self.self_attn(
            self.input_layernorm(hidden_states), 
            sin_table, 
            cos_table, 
            past_key_value
        )
        hidden_states += attn_output
        hidden_states += self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, current_key_value

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

    '''
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
    '''
    def forward(self, input_ids, past_key_values=None):
        hidden_states = self.embed_tokens(input_ids)
        batch_size, seq_len = input_ids.shape
        
        # 计算当前的 past_length (已经生成的长度)
        past_length = 0
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

        # 生成完整的 RoPE 表 (或者只生成需要的切片)
        # 这里为了简单，我们重新生成并切片
        total_length = past_length + seq_len
        sin_table, cos_table = generate_sin_and_cos_tables(
            total_length, self.head_dim, self.rope_theta, 
            hidden_states.dtype, hidden_states.device
        )
        # 只取当前位置对应的 RoPE (从 past_length 开始)
        sin_table = sin_table[past_length:]
        cos_table = cos_table[past_length:]

        next_cache = []
        for i in range(self.num_hidden_layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            hidden_states, cache = self.layers[i](
                hidden_states, sin_table, cos_table, past_kv
            )
            next_cache.append(cache)

        return self.norm(hidden_states), next_cache


class ModelForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.model = Model(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    '''
    def generate(self, input_ids, max_new_tokens=20):
        for _ in range(max_new_tokens):
            hidden_states = self.model(input_ids)

            logits = self.lm_head(hidden_states[:, -1, :])

            next = torch.argmax(logits, dim=-1).unsqueeze(-1)

            input_ids = torch.cat((input_ids, next), dim=-1)

        return input_ids
    '''
    def generate(self, input_ids, max_new_tokens=20):
        past_key_values = None
        
        for _ in range(max_new_tokens):
            # 如果是第一步(prefill)，使用完整的 input_ids
            # 如果是后续步骤(decode)，只使用最新的 token (形状 [Batch, 1])
            if past_key_values is None:
                model_inputs = input_ids
            else:
                model_inputs = input_ids[:, -1:]

            # 前向传播，获取 logits 和 新的 cache
            hidden_states, past_key_values = self.model(model_inputs, past_key_values)
            
            # 只需要最后一个 token 的 logits
            logits = self.lm_head(hidden_states[:, -1, :])
            next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
            
            # 拼接结果
            input_ids = torch.cat((input_ids, next_token), dim=-1)

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

        # 加速点六
        try:
            print("Compiling model with torch.compile...")
            model.model = torch.compile(model.model)
        except Exception as e:
            print(f"Compilation failed, falling back to eager mode: {e}")

        return model
