from __future__ import annotations

import io
import lzma
import os
import time

import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F

from train_gpt_test import (
    CastedLinear,
    GPT,
    Hyperparameters,
    _rebank_state_dict,
    _unbank_state_dict,
    build_sentencepiece_luts,
    dequantize_mixed_int6,
    load_validation_tokens,
    restore_low_dim_params_to_fp32,
)


def build_eval_model(args: Hyperparameters, device: torch.device) -> GPT:
    model = GPT(
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
        mtp_num_heads=0,
        mtp_loss_weight=0.0,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims,
        ln_scale=args.ln_scale,
        dtg=args.dtg_enabled,
        ve_enabled=args.ve_enabled,
        ve_dim=args.ve_dim,
        ve_layers=args.ve_layers,
        gated_attention=args.gated_attention,
        value_residual=args.value_residual,
    ).to(device).bfloat16()
    model.qo_bank.data = model.qo_bank.data.float()
    model.kv_bank.data = model.kv_bank.data.float()
    model.mlp_up_bank.data = model.mlp_up_bank.data.float()
    model.mlp_down_bank.data = model.mlp_down_bank.data.float()
    for module in model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(model)
    return model


def _snapshot_model_params(model: torch.nn.Module) -> list[torch.Tensor]:
    return [p.detach().clone() for p in model.parameters()]


def _assert_model_unchanged(model: torch.nn.Module, snapshots: list[torch.Tensor]) -> None:
    for idx, (param, snap) in enumerate(zip(model.parameters(), snapshots, strict=True)):
        if not torch.equal(param.detach(), snap):
            raise RuntimeError(f"Model parameter changed during MC dropout generation at index {idx}")
    if any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("Unexpected gradients found after MC dropout generation")


def mc_dropout_generate(
    model: GPT,
    sp: spm.SentencePieceProcessor,
    device: torch.device,
    prompt: str,
    steps: int,
    mc_passes: int,
    mc_dropout_p: float,
) -> tuple[list[int], list[int]]:
    if mc_passes <= 0:
        raise ValueError(f"MC_DROPOUT_PASSES must be > 0, got {mc_passes}")
    if not (0.0 <= mc_dropout_p < 1.0):
        raise ValueError(f"MC_DROPOUT_P must be in [0, 1), got {mc_dropout_p}")
    if steps < 0:
        raise ValueError(f"MC_GENERATE_STEPS must be >= 0, got {steps}")

    prompt_ids = sp.encode(prompt, out_type=int) if prompt else []
    if not prompt_ids:
        bos = int(sp.bos_id())
        prompt_ids = [bos if bos >= 0 else 0]

    tokens = torch.tensor([prompt_ids], dtype=torch.int64, device=device)
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        for _ in range(steps):
            probs_accum: torch.Tensor | None = None
            for _ in range(mc_passes):
                next_logits = model.forward_logits(tokens)[:, -1, :].float()
                if mc_dropout_p > 0.0:
                    next_logits = F.dropout(next_logits, p=mc_dropout_p, training=True)
                probs = torch.softmax(next_logits, dim=-1)
                probs_accum = probs if probs_accum is None else probs_accum + probs
            mean_probs = probs_accum / float(mc_passes)
            next_token = torch.argmax(mean_probs, dim=-1, keepdim=True).to(dtype=torch.int64)
            tokens = torch.cat((tokens, next_token), dim=1)
    if was_training:
        model.train()
    return tokens[0].tolist(), prompt_ids


def mc_eval_val(
    args: Hyperparameters,
    model: GPT,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: torch.Tensor,
    base_bytes_lut: torch.Tensor,
    has_leading_space_lut: torch.Tensor,
    is_boundary_token_lut: torch.Tensor,
    mc_passes: int,
    mc_dropout_p: float,
    eval_seq_len: int | None = None,
) -> tuple[float, float]:
    seq_len = eval_seq_len or args.train_seq_len
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, seq_len={seq_len}"
        )
    local_batch_seqs = local_batch_tokens // seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    was_training = model.training
    model.eval()
    compiled_logits = torch.compile(model.forward_logits, dynamic=False, fullgraph=True)
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * seq_len
            raw_end = batch_seq_end * seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)

            probs_accum = None
            for _ in range(mc_passes):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    logits = compiled_logits(x).float()
                if mc_dropout_p > 0.0:
                    logits = F.dropout(logits, p=mc_dropout_p, training=True)
                probs = torch.softmax(logits, dim=-1)
                probs_accum = probs if probs_accum is None else probs_accum + probs
            mean_probs = (probs_accum / float(mc_passes)).clamp_min(1e-9)
            nll = -torch.log(mean_probs.gather(-1, y.unsqueeze(-1)).squeeze(-1))
            loss_sum += nll.to(torch.float64).sum()
            batch_token_count = float(y.numel())
            token_count += batch_token_count

            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    val_loss = (loss_sum / token_count).item()
    bits_per_token = val_loss / torch.log(torch.tensor(2.0)).item()
    tokens_per_byte = token_count.item() / byte_count.item()
    if was_training:
        model.train()
    return val_loss, bits_per_token * tokens_per_byte


def mc_eval_val_sliding(
    args: Hyperparameters,
    model: GPT,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: torch.Tensor,
    base_bytes_lut: torch.Tensor,
    has_leading_space_lut: torch.Tensor,
    is_boundary_token_lut: torch.Tensor,
    stride: int,
    mc_passes: int,
    mc_dropout_p: float,
    batch_seqs: int = 32,
    eval_seq_len: int | None = None,
) -> tuple[float, float]:
    seq_len = eval_seq_len or args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, stride) if min(ws + seq_len, total_tokens) - ws >= 1]
    total_windows = len(window_starts)
    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    was_training = model.training
    model.eval()
    compiled_logits = torch.compile(model.forward_logits, dynamic=False, fullgraph=True)
    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi:bi + batch_seqs]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens: list[int] = []
            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]

            probs_accum = None
            for _ in range(mc_passes):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    logits = compiled_logits(x_batch).float()
                if mc_dropout_p > 0.0:
                    logits = F.dropout(logits, p=mc_dropout_p, training=True)
                probs = torch.softmax(logits, dim=-1)
                probs_accum = probs if probs_accum is None else probs_accum + probs
            mean_probs = (probs_accum / float(mc_passes)).clamp_min(1e-9)
            nll = -torch.log(mean_probs.gather(-1, y_batch.unsqueeze(-1)).squeeze(-1))

            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else max(wlen - stride, 0)
                scored_nll = nll[i, s:wlen].to(torch.float64)
                loss_sum += scored_nll.sum()
                token_count += float(wlen - s)
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    val_loss = (loss_sum / token_count).item()
    bits_per_token = val_loss / torch.log(torch.tensor(2.0)).item()
    tokens_per_byte = token_count.item() / byte_count.item()
    if was_training:
        model.train()
    return val_loss, bits_per_token * tokens_per_byte


def main() -> None:
    args = Hyperparameters()
    model_path = os.environ.get("MODEL_PATH", "final_model.int6.ptz")
    mc_dropout_passes = int(os.environ.get("MC_DROPOUT_PASSES", "32"))
    mc_dropout_p = float(os.environ.get("MC_DROPOUT_P", "0.01"))
    mc_generate_steps = int(os.environ.get("MC_GENERATE_STEPS", "64"))
    mc_prompt = os.environ.get("MC_PROMPT", "The")
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

    def log0(msg: str) -> None:
        if master_process:
            print(msg)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )

    effective_eval_seq_len = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    val_seq_len = max(args.train_seq_len, effective_eval_seq_len)
    val_tokens = load_validation_tokens(args.val_files, val_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    # Build a template state dict to recover dtypes/shapes for dequantization.
    cpu_template_model = build_eval_model(args, torch.device("cpu"))
    template_sd = {k: v.detach().cpu() for k, v in cpu_template_model.state_dict().items()}
    unbanked_template = _unbank_state_dict(template_sd, args.num_layers)
    del cpu_template_model

    with open(model_path, "rb") as f:
        quant_blob = f.read()
    quant_state = torch.load(io.BytesIO(lzma.decompress(quant_blob)), map_location="cpu")
    deq_unbanked = dequantize_mixed_int6(quant_state["w"], quant_state["m"], unbanked_template)
    deq_state = _rebank_state_dict(deq_unbanked, args.num_layers, template_sd)

    eval_model = build_eval_model(args, device)
    eval_model.load_state_dict(deq_state, strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = mc_eval_val(
        args,
        eval_model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        mc_passes=mc_dropout_passes,
        mc_dropout_p=mc_dropout_p,
        eval_seq_len=effective_eval_seq_len,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int6_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int6_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    sw_seq_len = effective_eval_seq_len
    if args.eval_stride > 0 and args.eval_stride < sw_seq_len:
        torch.cuda.synchronize()
        t_slide = time.perf_counter()
        sw_val_loss, sw_val_bpb = mc_eval_val_sliding(
            args,
            eval_model,
            rank,
            world_size,
            device,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            stride=args.eval_stride,
            mc_passes=mc_dropout_passes,
            mc_dropout_p=mc_dropout_p,
            eval_seq_len=sw_seq_len,
        )
        torch.cuda.synchronize()
        log0(
            f"final_int6_sliding_window val_loss:{sw_val_loss:.4f} val_bpb:{sw_val_bpb:.4f} "
            f"stride:{args.eval_stride} eval_time:{1000.0 * (time.perf_counter() - t_slide):.0f}ms "
            f"mc_passes:{mc_dropout_passes} mc_dropout_p:{mc_dropout_p:.4f}"
        )
        log0(f"final_int6_sliding_window_exact val_loss:{sw_val_loss:.8f} val_bpb:{sw_val_bpb:.8f}")
        log0(f"final_int8_zlib_roundtrip_exact val_loss:{sw_val_loss:.8f} val_bpb:{sw_val_bpb:.8f}")

    if args.eval_stride != 64 and 64 < sw_seq_len:
        torch.cuda.synchronize()
        t_slide64 = time.perf_counter()
        sw64_val_loss, sw64_val_bpb = mc_eval_val_sliding(
            args,
            eval_model,
            rank,
            world_size,
            device,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            stride=64,
            mc_passes=mc_dropout_passes,
            mc_dropout_p=mc_dropout_p,
            eval_seq_len=sw_seq_len,
        )
        torch.cuda.synchronize()
        log0(
            f"final_int6_sliding_window_s64 val_loss:{sw64_val_loss:.4f} val_bpb:{sw64_val_bpb:.4f} "
            f"stride:64 eval_time:{1000.0 * (time.perf_counter() - t_slide64):.0f}ms "
            f"mc_passes:{mc_dropout_passes} mc_dropout_p:{mc_dropout_p:.4f}"
        )
        log0(f"final_int6_sliding_window_s64_exact val_loss:{sw64_val_loss:.8f} val_bpb:{sw64_val_bpb:.8f}")
        log0(f"final_int8_zlib_roundtrip_exact val_loss:{sw64_val_loss:.8f} val_bpb:{sw64_val_bpb:.8f}")

    if master_process and mc_generate_steps > 0:
        weights_before = _snapshot_model_params(eval_model)
        t_mc = time.perf_counter()
        full_token_ids, prompt_token_ids = mc_dropout_generate(
            eval_model,
            sp,
            device,
            prompt=mc_prompt,
            steps=mc_generate_steps,
            mc_passes=mc_dropout_passes,
            mc_dropout_p=mc_dropout_p,
        )
        _assert_model_unchanged(eval_model, weights_before)
        generated_token_ids = full_token_ids[len(prompt_token_ids):]
        generated_text = sp.decode(generated_token_ids) if generated_token_ids else ""
        log0(
            f"mc_dropout_generation prompt_len:{len(prompt_token_ids)} "
            f"steps:{mc_generate_steps} passes:{mc_dropout_passes} p:{mc_dropout_p:.4f} "
            f"elapsed_ms:{1000.0 * (time.perf_counter() - t_mc):.0f}"
        )
        log0(f"mc_dropout_generation_text:{generated_text}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
