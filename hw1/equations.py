"""Формулы §5, §6: flops, memory, latency, energy от S и B (числа или массивы NumPy).

Соглашения:
- FLOPs: 1 MAC = 2 FLOPs, считаются только Conv и Linear (ReLU, пулинги и bias < 0.2 %).
- Bytes moved: без кешей, каждый кернел читает вход и веса из DRAM и пишет выход.
  FP32 = 4 байта, индексы MaxPool int64 = 8 байт.
- Memory: пик max_memory_allocated за один forward в inference_mode:
  веса + workspace cuBLAS + вход + максимум одновременно живых тензоров.
  Workspace cuDNN не учитывается, он зависит от выбранного алгоритма.
  У границы OOM cuDNN берёт алгоритмы без workspace, там формула точная.
"""
import numpy as np

F32 = 4
I64 = 8

# На CUDA max_pool2d идёт через max_pool2d_with_indices, индексы int64 пишутся и в inference_mode
MAXPOOL_INDICES = True

# Workspace cuBLAS выделяется при первом matmul и живёт до конца процесса:
# CUBLAS_WORKSPACE_CONFIG по умолчанию ":4096:2:16:8" = 8.125 MiB, плюс 1 MiB cuBLASLt
CUBLAS_WORKSPACE = int(9.125 * 2**20)

# При benchmark=False cuDNN считает conv2 через FFT с тайлингом, если B > 64
# (в профайлере около 1200 кернелов вместо 26)
CONV2_FFT_BATCH = 64

# имя, C_in, C_out, k, делитель S на входе, делитель S на выходе
CONVS = [
    ("conv1", 3, 32, 7, 1, 2),
    ("conv2", 32, 64, 5, 4, 4),
    ("conv3", 64, 128, 3, 4, 8),
    ("conv4", 128, 256, 1, 8, 8),
    ("conv5", 256, 256, 3, 8, 16),
    ("conv6", 256, 512, 1, 16, 16),
]
LINEARS = [("fc1", 512, 256), ("fc2", 256, 100)]  # у Linear bias есть

N_PARAMS = (sum(c_out * c_in * k * k for _, c_in, c_out, k, _, _ in CONVS)
            + sum(f_in * f_out + f_out for _, f_in, f_out in LINEARS))
WEIGHT_BYTES = F32 * N_PARAMS


def _grid(image_size, batch):
    s = np.asarray(image_size, dtype=float)
    b = np.asarray(batch, dtype=float)
    return np.broadcast_arrays(s, b)


def _out(x):
    return float(x) if np.ndim(x) == 0 else x


def op_costs(image_size, batch):
    """Список кернелов forward по порядку: (имя, тип, FLOPs, байты)."""
    S, B = _grid(image_size, batch)
    idx_bytes = I64 if MAXPOOL_INDICES else 0
    ops = []
    for name, c_in, c_out, k, d_in, d_out in CONVS:
        n_in = B * c_in * (S / d_in) ** 2
        n_out = B * c_out * (S / d_out) ** 2
        w = c_out * c_in * k * k
        ops.append((name, "conv", 2.0 * n_out * c_in * k * k, F32 * (n_in + w + n_out)))
        ops.append((f"relu_{name}", "relu", n_out, 2.0 * F32 * n_out))  # inplace: чтение + запись
        if name == "conv1":
            n_pool = B * 32 * (S / 4) ** 2
            ops.append(("maxpool", "maxpool", 9.0 * n_pool,
                        F32 * (n_out + n_pool) + idx_bytes * n_pool))
    n_in = B * 512 * (S / 16) ** 2
    ops.append(("avgpool", "avgpool", n_in, F32 * (n_in + B * 512)))
    for name, f_in, f_out in LINEARS:
        ops.append((name, "linear", 2.0 * B * f_in * f_out,
                    F32 * (B * f_in + f_in * f_out + f_out + B * f_out)))
        if name == "fc1":
            ops.append(("relu_fc1", "relu", B * f_out, 2.0 * F32 * B * f_out))
    return ops


def flops(image_size, batch):
    return _out(sum(f for _, kind, f, _ in op_costs(image_size, batch) if kind in ("conv", "linear")))


def bytes_moved(image_size, batch):
    """Трафик DRAM за forward в байтах, нужен для latency и energy."""
    return _out(sum(by for _, _, _, by in op_costs(image_size, batch)))


def memory(image_size, batch):
    """Пик памяти за forward в байтах."""
    S, B = _grid(image_size, batch)
    act = {name: F32 * B * c_out * (S / d_out) ** 2 for name, _, c_out, _, _, d_out in CONVS}
    pool = F32 * B * 32 * (S / 4) ** 2
    idx = I64 * B * 32 * (S / 4) ** 2 if MAXPOOL_INDICES else 0 * B
    x = F32 * B * 3 * S ** 2                 # вход держит вызывающий код весь forward
    live = [                                 # живые тензоры на каждом слое, кроме x и весов
        act["conv1"],                        # conv1, ReLU inplace
        act["conv1"] + pool + idx,           # maxpool: вход, выход, индексы
        pool + act["conv2"],
        act["conv2"] + act["conv3"],
        act["conv3"] + act["conv4"],
        act["conv4"] + act["conv5"],
        act["conv5"] + act["conv6"],
        act["conv6"] + F32 * B * 512,        # avgpool
        F32 * B * (512 + 256),               # fc1
        F32 * B * (256 + 100),               # fc2
    ]
    return _out(WEIGHT_BYTES + CUBLAS_WORKSPACE + x + np.maximum.reduce(live))


def _peak_flops(theta, name, kind, batch):
    p = theta["P"]
    if isinstance(p, dict):
        p = p.get(name, p.get(kind, p["default"]))
    if name == "conv2" and "P_conv2_fft" in theta:      # при больших B conv2 идёт через FFT
        p = np.where(batch > CONV2_FFT_BATCH, theta["P_conv2_fft"], p)
    return p


def latency_terms(image_size, batch, theta):
    """Для каждого кернела: (имя, тип, запуск, счёт, память) в секундах."""
    _, B = _grid(image_size, batch)
    return [(name, kind, np.full_like(f, theta["t_launch"]),
             f / _peak_flops(theta, name, kind, B), by / theta["bw"])
            for name, kind, f, by in op_costs(image_size, batch)]


def latency(image_size, batch, theta):
    """T = t0 + сумма по кернелам max(t_launch, FLOPs_k / P_k, Bytes_k / BW), секунды.

    Для conv2 при B > CONV2_FFT_BATCH вместо P_k берётся P_conv2_fft.
    """
    total = theta["t0"]
    for _, _, t_launch, t_compute, t_memory in latency_terms(image_size, batch, theta):
        total = total + np.maximum(t_launch, np.maximum(t_compute, t_memory))
    return _out(total)


def energy(image_size, batch, theta_energy):
    """E = P_static * T + e_flop * FLOPs + e_byte * Bytes, джоули."""
    t = latency(image_size, batch, theta_energy["latency_theta"])
    return _out(theta_energy["P_static"] * t
                + theta_energy["e_flop"] * flops(image_size, batch)
                + theta_energy["e_byte"] * bytes_moved(image_size, batch))
