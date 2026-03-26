import torch

import train_gpt_turboquant as tq


def _unit_vector(dim: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(dim, generator=g, dtype=torch.float32)
    return x / x.norm().clamp_min(1e-12)


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
