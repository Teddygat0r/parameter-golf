import math

import torch

import train_gpt_turboquant as tq


def _unit_vector(dim: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(dim, generator=g, dtype=torch.float32)
    return x / x.norm().clamp_min(1e-12)


def _random_matrix(rows: int, cols: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(rows, cols, generator=g, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Original tests (preserved)
# ---------------------------------------------------------------------------


def test_turboquant_mse_roundtrip_and_bitwidth_trend() -> None:
    dim = 256
    x = _unit_vector(dim, seed=1)
    q1 = tq.TurboQuantMSE(dim=dim, bits=1, rotation_seed=7)
    q3 = tq.TurboQuantMSE(dim=dim, bits=3, rotation_seed=7)

    idx1 = q1.quantize(x)
    idx3 = q3.quantize(x)
    xhat1 = q1.dequantize(idx1)
    xhat3 = q3.dequantize(idx3)

    assert xhat1.shape == x.shape
    assert xhat3.shape == x.shape
    assert torch.isfinite(xhat1).all()
    assert torch.isfinite(xhat3).all()
    mse1 = torch.mean((x - xhat1) ** 2).item()
    mse3 = torch.mean((x - xhat3) ** 2).item()
    assert mse3 < mse1


def test_turboquant_encoding_is_deterministic_for_fixed_seed() -> None:
    dim = 192
    x = _unit_vector(dim, seed=2)
    q_mse_a = tq.TurboQuantMSE(dim=dim, bits=2, rotation_seed=11)
    q_mse_b = tq.TurboQuantMSE(dim=dim, bits=2, rotation_seed=11)
    idx_a = q_mse_a.quantize(x)
    idx_b = q_mse_b.quantize(x)
    assert torch.equal(idx_a, idx_b)

    q_prod_a = tq.TurboQuantProd(dim=dim, bits=2, rotation_seed=11, qjl_seed=123)
    q_prod_b = tq.TurboQuantProd(dim=dim, bits=2, rotation_seed=11, qjl_seed=123)
    payload_a = q_prod_a.quantize(x)
    payload_b = q_prod_b.quantize(x)
    assert torch.equal(payload_a["idx"], payload_b["idx"])
    assert torch.equal(payload_a["qjl"], payload_b["qjl"])
    assert abs(float(payload_a["gamma"]) - float(payload_b["gamma"])) < 1e-7


def test_turboquant_prod_is_statistically_unbiased_for_inner_product() -> None:
    dim = 256
    bits = 3
    x = _unit_vector(dim, seed=3)
    y = _unit_vector(dim, seed=4)
    target = torch.dot(x, y).item()

    estimates = []
    for s in range(64):
        q = tq.TurboQuantProd(dim=dim, bits=bits, rotation_seed=17, qjl_seed=1_000 + s)
        payload = q.quantize(x)
        estimates.append(q.estimate_inner_product(y, payload))
    mean_est = float(sum(estimates) / len(estimates))
    assert abs(mean_est - target) < 0.05


def test_turboquant_prod_reduces_low_bit_inner_product_bias_vs_mse() -> None:
    dim = 256
    bits = 2
    x = _unit_vector(dim, seed=5)
    y = _unit_vector(dim, seed=6)
    target = torch.dot(x, y).item()

    q_mse = tq.TurboQuantMSE(dim=dim, bits=bits, rotation_seed=29)
    xhat_mse = q_mse.dequantize(q_mse.quantize(x))
    mse_est = torch.dot(y, xhat_mse).item()
    mse_bias = abs(mse_est - target)

    prod_estimates = []
    for s in range(64):
        q_prod = tq.TurboQuantProd(dim=dim, bits=bits, rotation_seed=29, qjl_seed=2_000 + s)
        payload = q_prod.quantize(x)
        prod_estimates.append(q_prod.estimate_inner_product(y, payload))
    prod_mean = float(sum(prod_estimates) / len(prod_estimates))
    prod_bias = abs(prod_mean - target)
    assert prod_bias < mse_bias


def test_turboquant_state_dict_serialization_roundtrip() -> None:
    g = torch.Generator(device="cpu").manual_seed(77)
    state = {
        "linear.weight": torch.randn(64, 32, generator=g, dtype=torch.float32),
        "linear.bias": torch.randn(64, generator=g, dtype=torch.float32),
        "count": torch.tensor([1, 2, 3], dtype=torch.int64),
    }
    obj, stats = tq.quantize_state_dict_turboquant(state, bits=3, mode="prod", rotation_seed=101, qjl_seed=202)
    assert stats["num_tensors"] == len(state)
    restored = tq.dequantize_state_dict_turboquant(obj)

    assert set(restored) == set(state)
    assert restored["linear.weight"].shape == state["linear.weight"].shape
    assert restored["linear.bias"].shape == state["linear.bias"].shape
    assert restored["count"].dtype == torch.int64
    assert torch.equal(restored["count"], state["count"])
    assert torch.isfinite(restored["linear.weight"]).all()
    assert torch.isfinite(restored["linear.bias"]).all()


# ---------------------------------------------------------------------------
# New robust correctness tests
# ---------------------------------------------------------------------------

# Paper's Theorem 1 bounds for D_mse on unit vectors (||x||=1).
_PAPER_MSE_BOUNDS: dict[int, float] = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}


def test_rotation_matrix_is_orthogonal() -> None:
    """Q from _build_rotation_matrix must satisfy Q^T Q = I."""
    for dim in [64, 256, 512]:
        q = tq._build_rotation_matrix(dim, seed=42)
        assert q is not None
        eye = torch.eye(dim)
        assert torch.allclose(q.T @ q, eye, atol=1e-5), (
            f"Q^T Q != I for dim={dim}, max error={torch.max(torch.abs(q.T @ q - eye)).item()}"
        )
        assert torch.allclose(q @ q.T, eye, atol=1e-5)


def test_codebook_is_sorted_and_symmetric() -> None:
    """Lloyd-Max codebook must be sorted and approximately symmetric about 0."""
    for dim in [128, 256, 512]:
        for bits in [1, 2, 3, 4]:
            cb = tq._build_lloyd_max_codebook_for_sphere_coord(dim, bits)
            assert cb.shape == (1 << bits,)
            diffs = cb[1:] - cb[:-1]
            assert (diffs > 0).all(), f"Codebook not sorted for dim={dim}, bits={bits}"
            assert torch.allclose(cb, -cb.flip(0), atol=1e-5), (
                f"Codebook not symmetric for dim={dim}, bits={bits}"
            )


def test_mse_quantizer_error_within_paper_bounds() -> None:
    """
    Theorem 1: For ||x||=1, D_mse(b) ≈ {1:0.36, 2:0.117, 3:0.03, 4:0.009}.
    Average MSE over many random unit vectors must be within 2x of the bound.
    """
    dim = 256
    n_samples = 100
    for bits, paper_bound in _PAPER_MSE_BOUNDS.items():
        q = tq.TurboQuantMSE(dim=dim, bits=bits, rotation_seed=7)
        total_mse = 0.0
        for seed in range(n_samples):
            x = _unit_vector(dim, seed=seed + 1000)
            idx = q.quantize(x)
            x_hat = q.dequantize(idx)
            total_mse += (x - x_hat).square().sum().item()
        avg_mse = total_mse / n_samples
        assert avg_mse < paper_bound * 2.0, (
            f"bits={bits}: avg MSE {avg_mse:.4f} exceeds 2x paper bound {paper_bound}"
        )
        assert avg_mse > paper_bound * 0.5, (
            f"bits={bits}: avg MSE {avg_mse:.4f} suspiciously below 0.5x paper bound"
        )


def test_mse_quantizer_preserves_direction() -> None:
    """Cosine similarity between x and dequantized x̂ must be high for bits >= 2."""
    dim = 512
    for bits, min_cos in [(2, 0.90), (3, 0.96), (4, 0.99)]:
        q = tq.TurboQuantMSE(dim=dim, bits=bits, rotation_seed=13)
        cos_sims = []
        for seed in range(50):
            x = _unit_vector(dim, seed=seed + 2000)
            x_hat = q.dequantize(q.quantize(x))
            cs = torch.dot(x, x_hat) / (x.norm() * x_hat.norm())
            cos_sims.append(cs.item())
        avg_cos = sum(cos_sims) / len(cos_sims)
        assert avg_cos > min_cos, (
            f"bits={bits}: avg cos_sim {avg_cos:.4f} < {min_cos}"
        )


def test_mse_mode_gives_strictly_lower_reconstruction_error_than_prod() -> None:
    """
    For weight quantization, MSE mode must reconstruct weights more faithfully
    than prod mode at the same bit budget. Prod mode wastes one bit on QJL.
    """
    dim = 256
    bits = 3
    n_samples = 50
    mse_errors, prod_errors = [], []
    for seed in range(n_samples):
        x = _unit_vector(dim, seed=seed + 3000)
        q_mse = tq.TurboQuantMSE(dim=dim, bits=bits, rotation_seed=7)
        x_hat_mse = q_mse.dequantize(q_mse.quantize(x))
        mse_errors.append((x - x_hat_mse).square().sum().item())

        q_prod = tq.TurboQuantProd(dim=dim, bits=bits, rotation_seed=7, qjl_seed=seed)
        payload = q_prod.quantize(x)
        x_hat_prod = q_prod.dequantize(payload)
        prod_errors.append((x - x_hat_prod).square().sum().item())

    avg_mse = sum(mse_errors) / len(mse_errors)
    avg_prod = sum(prod_errors) / len(prod_errors)
    assert avg_mse < avg_prod, (
        f"MSE mode ({avg_mse:.4f}) should have lower error than prod ({avg_prod:.4f})"
    )
    assert avg_prod > 3.0 * avg_mse, (
        f"Prod MSE ({avg_prod:.4f}) should be significantly (>3x) worse than MSE ({avg_mse:.4f})"
    )


def test_default_quantization_mode_is_mse() -> None:
    """
    Weight quantization must use MSE mode by default. Prod mode is designed for
    KV cache / inner-product tasks and adds ~5x more reconstruction error.
    """
    assert tq.TURBOQUANT_DEFAULT_MODE == "mse", (
        f"Default mode is '{tq.TURBOQUANT_DEFAULT_MODE}', expected 'mse'. "
        "Prod mode is for KV cache compression, not weight quantization."
    )

def test_idx_bitpack_roundtrip_and_size() -> None:
    rows, dim, bits = 17, 31, 5
    levels = 1 << bits
    idx = torch.arange(rows * dim, dtype=torch.int16).reshape(rows, dim).remainder(levels)
    packed = tq._pack_indices_to_uint8(idx, bits=bits)
    restored = tq._unpack_indices_from_uint8(packed, bits=bits, rows=rows, dim=dim)
    assert torch.equal(restored, idx)
    expected_bytes = (rows * dim * bits + 7) // 8
    assert int(packed.numel()) == expected_bytes


def test_state_dict_idx_storage_uses_packed_bytes() -> None:
    bits = 5
    threshold = tq.TURBOQUANT_KEEP_FLOAT_MAX_NUMEL
    # Must exceed keep-float threshold so the tensor is actually quantized.
    rows, dim = 257, 257
    assert rows * dim > threshold
    block_size = 128
    state = {
        "w": _random_matrix(rows, dim, seed=1234),
    }
    obj, _ = tq.quantize_state_dict_turboquant(
        state, bits=bits, mode="mse", block_size=block_size, rotation_seed=0, qjl_seed=0
    )
    meta = obj["quant_meta"]["w"]
    assert meta["blockwise"] is True
    assert meta["block_size"] == block_size
    block_dims = [int(v) for v in meta["block_dims"]]
    assert block_dims == [128, 128, 1]
    total_bytes = 0
    for i, block_dim in enumerate(block_dims):
        packed_idx = obj["quantized"][f"w::idx::{i}"]
        assert packed_idx.dtype == torch.uint8
        expected_block_bytes = (rows * block_dim * bits + 7) // 8
        assert int(packed_idx.numel()) == expected_block_bytes
        total_bytes += expected_block_bytes
    assert total_bytes == ((rows * dim * bits + 7) // 8)
    assert obj["quant_meta"]["w"]["idx_packed"] is True


def test_state_dict_roundtrip_large_tensors_mse_mode() -> None:
    """
    State dict roundtrip with MSE mode on tensors large enough to be quantized
    (numel > TURBOQUANT_KEEP_FLOAT_MAX_NUMEL) must preserve weights accurately.
    """
    threshold = tq.TURBOQUANT_KEEP_FLOAT_MAX_NUMEL
    state = {
        "attn.weight": _random_matrix(512, 512, seed=100),
        "mlp.weight": _random_matrix(1024, 512, seed=101),
        "small_param": torch.randn(32, dtype=torch.float32),
        "int_tensor": torch.tensor([10, 20, 30], dtype=torch.int64),
    }
    assert state["attn.weight"].numel() > threshold
    assert state["mlp.weight"].numel() > threshold

    obj, stats = tq.quantize_state_dict_turboquant(
        state, bits=3, mode="mse", block_size=128, rotation_seed=42, qjl_seed=43
    )
    restored = tq.dequantize_state_dict_turboquant(obj)

    assert set(restored) == set(state)
    assert torch.equal(restored["int_tensor"], state["int_tensor"])

    for name in ["attn.weight", "mlp.weight"]:
        orig = state[name].float()
        rest = restored[name].float()
        assert rest.shape == orig.shape
        assert torch.isfinite(rest).all()

        rel_err = (orig - rest).norm().item() / orig.norm().item()
        assert rel_err < 0.25, (
            f"{name}: relative Frobenius error {rel_err:.4f} >= 0.25 with MSE mode"
        )

        cos_sims = []
        for i in range(orig.shape[0]):
            cs = torch.dot(orig[i], rest[i]) / (orig[i].norm() * rest[i].norm() + 1e-12)
            cos_sims.append(cs.item())
        avg_cos = sum(cos_sims) / len(cos_sims)
        assert avg_cos > 0.97, (
            f"{name}: avg row cosine similarity {avg_cos:.4f} < 0.97 with MSE mode"
        )


def test_blockwise_mse_roundtrip_with_tail_block() -> None:
    rows, dim = 600, 130
    bits = 3
    block_size = 128
    state = {"w": _random_matrix(rows, dim, seed=888)}
    obj, _ = tq.quantize_state_dict_turboquant(
        state,
        bits=bits,
        mode="mse",
        block_size=block_size,
        rotation_seed=123,
        qjl_seed=456,
    )
    meta = obj["quant_meta"]["w"]
    assert bool(meta.get("blockwise", False))
    assert int(meta["num_blocks"]) == 2
    assert [int(v) for v in meta["block_dims"]] == [128, 2]
    restored = tq.dequantize_state_dict_turboquant(obj)["w"].float()
    original = state["w"].float()
    rel_err = (original - restored).norm().item() / original.norm().item()
    assert rel_err < 0.3


def test_blockwise_mse_serialization_stores_seeds_not_rotation_matrix() -> None:
    state = {"w": _random_matrix(260, 260, seed=999)}
    obj, _ = tq.quantize_state_dict_turboquant(
        state,
        bits=3,
        mode="mse",
        block_size=128,
        rotation_seed=17,
        qjl_seed=18,
    )
    meta = obj["quant_meta"]["w"]
    assert "rotation_seed" in meta
    assert "block_rotation_seeds" in meta
    quantized_keys = set(obj["quantized"].keys())
    assert not any("rotation" in key.lower() for key in quantized_keys)
    assert not any("r_matrix" in key.lower() for key in quantized_keys)


def test_state_dict_roundtrip_preserves_shapes_and_dtypes() -> None:
    """All tensors must retain their original shape and floating-point dtype."""
    state = {
        "w1": _random_matrix(512, 256, seed=200).bfloat16(),
        "w2": _random_matrix(256, 512, seed=201),
        "bias": torch.randn(256, dtype=torch.float32),
        "counts": torch.arange(10, dtype=torch.int32),
    }
    for mode in ["mse", "prod"]:
        obj, _ = tq.quantize_state_dict_turboquant(
            state, bits=3, mode=mode, rotation_seed=50, qjl_seed=51
        )
        restored = tq.dequantize_state_dict_turboquant(obj)
        for name, orig_t in state.items():
            rest_t = restored[name]
            assert rest_t.shape == orig_t.shape, (
                f"mode={mode}, {name}: shape {rest_t.shape} != {orig_t.shape}"
            )
            if not orig_t.is_floating_point():
                assert rest_t.dtype == orig_t.dtype
                assert torch.equal(rest_t, orig_t)


def test_mse_quantizer_roundtrip_is_exact_inverse_of_rotation() -> None:
    """Quantize→dequantize must apply rotation, quantize, then un-rotate correctly."""
    dim = 128
    bits = 2
    q = tq.TurboQuantMSE(dim=dim, bits=bits, rotation_seed=99)
    x = _unit_vector(dim, seed=500)
    idx = q.quantize(x)
    x_hat = q.dequantize(idx)

    y_rotated = q._rotate(x.unsqueeze(0))[0]
    y_hat_from_codebook = q.codebook[idx.long()]
    x_hat_manual = q._unrotate(y_hat_from_codebook.unsqueeze(0))[0]
    assert torch.allclose(x_hat, x_hat_manual, atol=1e-6), (
        "dequantize path doesn't match manual rotate→codebook→unrotate"
    )


def test_prod_quantizer_qjl_is_unbiased_for_residual() -> None:
    """
    The QJL correction must be an unbiased estimate of the residual:
    E[qjl_part] ≈ r = x - x_mse.
    """
    dim = 256
    bits = 3
    x = _unit_vector(dim, seed=600)
    q_mse = tq.TurboQuantMSE(dim=dim, bits=bits - 1, rotation_seed=7)
    idx = q_mse.quantize(x)
    x_mse = q_mse.dequantize(idx)
    residual = x - x_mse

    qjl_parts = []
    for s in range(200):
        q_prod = tq.TurboQuantProd(dim=dim, bits=bits, rotation_seed=7, qjl_seed=5000 + s)
        payload = q_prod.quantize(x)
        x_hat = q_prod.dequantize(payload)
        qjl_parts.append(x_hat - x_mse)

    avg_qjl = torch.stack(qjl_parts).mean(dim=0)
    bias = (avg_qjl - residual).norm().item()
    residual_norm = residual.norm().item()
    assert bias < 0.15 * residual_norm, (
        f"QJL bias {bias:.4f} exceeds 15% of residual norm {residual_norm:.4f}"
    )


def test_batch_quantize_matches_single() -> None:
    """Batch and single-vector quantize/dequantize must produce identical results."""
    dim = 128
    bits = 3
    q_mse = tq.TurboQuantMSE(dim=dim, bits=bits, rotation_seed=11)
    q_prod = tq.TurboQuantProd(dim=dim, bits=bits, rotation_seed=11, qjl_seed=22)

    vecs = [_unit_vector(dim, seed=700 + i) for i in range(8)]
    batch = torch.stack(vecs)

    batch_idx = q_mse.quantize_batch(batch)
    batch_hat = q_mse.dequantize_batch(batch_idx)
    for i, v in enumerate(vecs):
        single_idx = q_mse.quantize(v)
        single_hat = q_mse.dequantize(single_idx)
        assert torch.equal(batch_idx[i], single_idx)
        assert torch.allclose(batch_hat[i], single_hat, atol=1e-6)

    batch_payload = q_prod.quantize_batch(batch)
    batch_hat_prod = q_prod.dequantize_batch(batch_payload)
    for i, v in enumerate(vecs):
        single_payload = q_prod.quantize(v)
        single_hat = q_prod.dequantize(single_payload)
        assert torch.equal(batch_payload["idx"][i], single_payload["idx"])
        assert torch.equal(batch_payload["qjl"][i], single_payload["qjl"])
        assert torch.allclose(batch_hat_prod[i], single_hat, atol=1e-6)


def test_higher_bits_always_gives_lower_mse() -> None:
    """MSE must strictly decrease as bits increases for every bit width pair."""
    dim = 256
    n_samples = 30
    bit_range = [1, 2, 3, 4]
    avg_mse_by_bits: dict[int, float] = {}
    for bits in bit_range:
        q = tq.TurboQuantMSE(dim=dim, bits=bits, rotation_seed=7)
        total = 0.0
        for seed in range(n_samples):
            x = _unit_vector(dim, seed=seed + 800)
            x_hat = q.dequantize(q.quantize(x))
            total += (x - x_hat).square().sum().item()
        avg_mse_by_bits[bits] = total / n_samples

    for b in range(1, len(bit_range)):
        lo, hi = bit_range[b - 1], bit_range[b]
        assert avg_mse_by_bits[hi] < avg_mse_by_bits[lo], (
            f"MSE did not decrease: bits={lo} -> {avg_mse_by_bits[lo]:.4f}, "
            f"bits={hi} -> {avg_mse_by_bits[hi]:.4f}"
        )


def test_zero_vector_roundtrip() -> None:
    """Quantizing and dequantizing a zero vector should return near-zero."""
    dim = 128
    x = torch.zeros(dim)
    q = tq.TurboQuantMSE(dim=dim, bits=3, rotation_seed=0)
    idx = q.quantize(x)
    x_hat = q.dequantize(idx)
    assert torch.isfinite(x_hat).all()

    state = {"w": torch.zeros(256, dim)}
    obj, _ = tq.quantize_state_dict_turboquant(
        state, bits=3, mode="mse", rotation_seed=0, qjl_seed=0
    )
    restored = tq.dequantize_state_dict_turboquant(obj)
    assert torch.allclose(restored["w"], torch.zeros_like(restored["w"]), atol=1e-5)


def test_quantize_dequantize_does_not_mutate_codebook_or_rotation() -> None:
    """Cached codebook and rotation matrices must not be modified in-place."""
    dim = 128
    bits = 2
    q = tq.TurboQuantMSE(dim=dim, bits=bits, rotation_seed=55)
    cb_before = q.codebook.clone()
    rot_before = q.rotation.clone() if q.rotation is not None else None

    for seed in range(20):
        x = _unit_vector(dim, seed=seed + 900)
        idx = q.quantize(x)
        _ = q.dequantize(idx)

    assert torch.equal(q.codebook, cb_before)
    if rot_before is not None:
        assert torch.equal(q.rotation, rot_before)
