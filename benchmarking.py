from __future__ import annotations

import io
import os
import random
import time
import zlib

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

from train_gpt import (
    CastedLinear,
    CONTROL_TENSOR_NAME_PATTERNS,
    GPT,
    Hyperparameters,
    build_sentencepiece_luts,
    eval_val,
    load_validation_tokens,
    restore_low_dim_params_to_fp32,
)
from train_gpt_turboquant import (
    _build_lloyd_max_codebook_for_sphere_coord,
    _build_qjl_matrix,
    dequantize_state_dict_turboquant,
    quantize_state_dict_turboquant,
)


NBIT_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "NBIT_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
NBIT_KEEP_FLOAT_MAX_NUMEL = 65_536
NBIT_KEEP_FLOAT_STORE_DTYPE = torch.float16
NBIT_PER_ROW_SCALE_DTYPE = torch.float16
NBIT_CLIP_PERCENTILE = float(os.environ.get("NBIT_CLIP_PERCENTILE", 99.99984))
NBIT_CLIP_Q = NBIT_CLIP_PERCENTILE / 100.0


def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def keep_float_tensor_nbit(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in NBIT_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=NBIT_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t


def _pack_unsigned_values(vals: Tensor, bits: int) -> Tensor:
    flat = vals.to(torch.uint8).reshape(-1).cpu().numpy().astype(np.uint16, copy=False)
    bit_positions = np.arange(bits, dtype=np.uint16)
    bit_matrix = ((flat[:, None] >> bit_positions[None, :]) & 1).astype(np.uint8, copy=False)
    packed = np.packbits(bit_matrix.reshape(-1), bitorder="little")
    return torch.from_numpy(packed.copy())


def _unpack_unsigned_values(packed: Tensor, bits: int, numel: int) -> Tensor:
    packed_np = packed.to(torch.uint8).reshape(-1).cpu().numpy()
    flat_bits = np.unpackbits(packed_np, bitorder="little")
    needed_bits = numel * bits
    bits_mat = flat_bits[:needed_bits].reshape(numel, bits).astype(np.uint16, copy=False)
    vals = (bits_mat << np.arange(bits, dtype=np.uint16)[None, :]).sum(axis=1).astype(np.uint8, copy=False)
    return torch.from_numpy(vals.copy())


def quantize_float_tensor_nbit(t: Tensor, bits: int) -> tuple[Tensor, Tensor]:
    q_pos = (1 << (bits - 1)) - 1
    q_neg = -(1 << (bits - 1))
    unsigned_offset = 1 << (bits - 1)
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), NBIT_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / float(q_pos)).clamp_min(1.0 / float(max(q_pos, 1)))
        q = torch.clamp(torch.round(clipped / scale[:, None]), q_neg, q_pos).to(torch.int16)
        q_unsigned = (q + unsigned_offset).to(torch.uint8)
        packed = _pack_unsigned_values(q_unsigned, bits=bits)
        return packed, scale.to(dtype=NBIT_PER_ROW_SCALE_DTYPE).contiguous()

    clip_abs = float(torch.quantile(t32.abs().flatten(), NBIT_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / float(q_pos) if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), q_neg, q_pos).to(torch.int16)
    q_unsigned = (q + unsigned_offset).to(torch.uint8)
    packed = _pack_unsigned_values(q_unsigned, bits=bits)
    return packed, scale


def quantize_state_dict_nbit(state_dict: dict[str, Tensor], bits: int):
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "nbit_payload_bytes"),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["nbit_payload_bytes"] += tensor_nbytes(t)
            continue

        if t.numel() <= NBIT_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor_nbit(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["nbit_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        q_packed, s = quantize_float_tensor_nbit(t, bits=bits)
        quantized[name] = q_packed
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        shapes[name] = tuple(t.shape)
        qmeta[name] = {
            "scheme": "nbit_packed_per_row" if t.ndim == 2 else "nbit_packed_per_tensor",
            "bits": bits,
        }
        stats["nbit_payload_bytes"] += tensor_nbytes(q_packed) + tensor_nbytes(s)

    obj: dict[str, object] = {
        "__quant_format__": "nbit_clean_packed_v1",
        "bits": bits,
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "shapes": shapes,
        "passthrough": passthrough,
        "qmeta": qmeta,
    }
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict_nbit(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    bits = int(obj["bits"])
    unsigned_offset = 1 << (bits - 1)
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, packed in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        shape = tuple(obj["shapes"][name])
        s = obj["scales"][name]
        numel = int(np.prod(shape))
        unpacked_unsigned = _unpack_unsigned_values(packed, bits=bits, numel=numel)
        q = unpacked_unsigned.to(torch.int16).sub_(unsigned_offset).to(torch.float32).reshape(shape)
        if s.ndim > 0:
            scale = s.to(dtype=torch.float32)
            out[name] = (q * scale.view(shape[0], *([1] * (len(shape) - 1)))).to(dtype=dtype).contiguous()
        else:
            out[name] = (q * float(s.item())).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


def estimate_turboquant_auxiliary_bytes(quant_obj: dict[str, object]) -> tuple[int, dict[str, int]]:
    quant_meta = quant_obj.get("quant_meta", {})
    if not isinstance(quant_meta, dict):
        return 0, {"codebook_bytes": 0, "qjl_bytes": 0}

    codebook_keys: set[tuple[int, int]] = set()
    qjl_keys: set[tuple[int, int]] = set()
    for meta in quant_meta.values():
        if not isinstance(meta, dict):
            continue
        bits = int(meta["bits"])
        kind = str(meta["kind"])
        if kind == "mse" and bool(meta.get("blockwise", False)):
            block_dims = [int(v) for v in meta.get("block_dims", [])]
            if not block_dims:
                continue
            for block_dim in block_dims:
                codebook_keys.add((block_dim, bits))
        else:
            dim = int(meta["dim"])
            codebook_bits = bits if kind == "mse" else bits - 1
            codebook_keys.add((dim, codebook_bits))
        if kind == "prod":
            dim = int(meta["dim"])
            qjl_keys.add((dim, int(meta["qjl_seed"])))

    codebook_bytes = sum(tensor_nbytes(_build_lloyd_max_codebook_for_sphere_coord(dim=dim, bits=bits)) for dim, bits in codebook_keys)
    qjl_bytes = sum(tensor_nbytes(_build_qjl_matrix(dim=dim, seed=seed)) for dim, seed in qjl_keys)

    total = codebook_bytes + qjl_bytes
    return total, {
        "codebook_bytes": codebook_bytes,
        "qjl_bytes": qjl_bytes,
    }


def run_eval(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float, float]:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    val_loss, val_bpb = eval_val(
        args,
        model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    eval_ms = 1000.0 * (time.perf_counter() - t0)
    return val_loss, val_bpb, eval_ms


def main() -> None:
    args = Hyperparameters()
    quant_bits = int(os.environ.get("QUANT_BITS", "6"))
    quant_block_size = int(os.environ.get("TURBOQUANT_BLOCK_SIZE", "16"))
    quant_method = os.environ.get("QUANT_METHOD", "naive").lower()
    quant_mode = os.environ.get("TURBOQUANT_MODE", "mse").lower()
    if quant_method not in {"naive", "turboquant"}:
        raise ValueError(f"QUANT_METHOD must be one of {{'naive', 'turboquant'}}, got {quant_method}")
    if quant_method == "naive":
        if quant_bits < 2 or quant_bits > 8:
            raise ValueError(f"QUANT_BITS must be in [2, 8] for QUANT_METHOD=naive, got {quant_bits}")
    else:
        if quant_mode not in {"mse", "prod"}:
            raise ValueError(f"TURBOQUANT_MODE must be one of {{'mse', 'prod'}}, got {quant_mode}")
        min_bits = 2 if quant_mode == "prod" else 1
        if quant_bits < min_bits or quant_bits > 8:
            raise ValueError(
                f"QUANT_BITS must be in [{min_bits}, 8] for TURBOQUANT_MODE={quant_mode}, got {quant_bits}"
            )

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)

    state_path = os.environ.get("MODEL_PATH", "final_model.pt")
    state = torch.load(state_path, map_location="cpu")
    base_model.load_state_dict(state, strict=True)

    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    # val_loss, val_bpb, eval_ms = run_eval(
    #     args,
    #     model,
    #     rank,
    #     world_size,
    #     device,
    #     grad_accum_steps,
    #     val_tokens,
    #     base_bytes_lut,
    #     has_leading_space_lut,
    #     is_boundary_token_lut,
    # )

    # if master_process:
    #     print(f"benchmark_final_pt val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} eval_time:{eval_ms:.0f}ms")
    #     print(f"benchmark_final_pt_exact val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f}")

    if quant_method == "naive":
        quant_obj, quant_stats = quantize_state_dict_nbit(base_model.state_dict(), bits=quant_bits)
        payload_bytes = int(quant_stats["nbit_payload_bytes"])
        ratio = quant_stats["baseline_tensor_bytes"] / max(payload_bytes, 1)
        quant_tag = f"int{quant_bits}"
        nbit_path = os.environ.get("NBIT_MODEL_PATH", f"final_model.int{quant_bits}.ptz")
    else:
        quant_obj, quant_stats = quantize_state_dict_turboquant(
            base_model.state_dict(),
            bits=quant_bits,
            mode=quant_mode,
            block_size=quant_block_size,
            rotation_seed=args.seed,
            qjl_seed=args.seed,
        )
        payload_bytes = int(quant_stats["turboquant_payload_bytes"])
        ratio = quant_stats["baseline_tensor_bytes"] / max(payload_bytes, 1)
        quant_tag = f"turboquant_{quant_mode}_int{quant_bits}"
        nbit_path = os.environ.get("NBIT_MODEL_PATH", f"final_model.turboquant.{quant_mode}.int{quant_bits}.ptz")

    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_raw_bytes = len(quant_raw)
    if master_process:
        with open(nbit_path, "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize(nbit_path)
        print(
            f"serialized_model_{quant_tag}_zlib bytes:{quant_file_bytes} "
            f"(payload:{payload_bytes} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        if quant_method == "turboquant":
            aux_bytes, aux_breakdown = estimate_turboquant_auxiliary_bytes(quant_obj)
            effective_bytes = quant_file_bytes + aux_bytes
            print(
                f"effective_turboquant_artifact_bytes:{effective_bytes} "
                f"(zlib:{quant_file_bytes} codebook:{aux_breakdown['codebook_bytes']} "
                f"qjl:{aux_breakdown['qjl_bytes']})"
            )

    if distributed:
        dist.barrier()

    with open(nbit_path, "rb") as f:
        quant_blob_disk = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")
    if quant_method == "naive":
        base_model.load_state_dict(dequantize_state_dict_nbit(quant_state), strict=True)
    else:
        base_model.load_state_dict(dequantize_state_dict_turboquant(quant_state), strict=True)

    q_val_loss, q_val_bpb, q_eval_ms = run_eval(
        args,
        model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    if master_process:
        print(
            f"benchmark_final_{quant_tag}_zlib_roundtrip val_loss:{q_val_loss:.4f} "
            f"val_bpb:{q_val_bpb:.4f} eval_time:{q_eval_ms:.0f}ms"
        )
        print(
            f"benchmark_final_{quant_tag}_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} "
            f"val_bpb:{q_val_bpb:.8f}"
        )

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
