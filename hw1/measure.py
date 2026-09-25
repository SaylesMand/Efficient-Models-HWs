"""Замеры latency, пика памяти и энергии одного forward на сетке (S, B).

Протокол (§5, §7):
- eval(), inference_mode, FP32, cudnn.benchmark и TF32 выключены;
- latency: медиана по CUDA events после прогрева;
- memory: max_memory_allocated за один forward;
- energy: счётчик NVML всего GPU за K прогонов подряд, делённый на K.
Вход - случайные тензоры, при нехватке памяти пишется OOM.

Запуск на T4 (около 25 мин): python measure.py
Пишет results/measurements.csv и results/figures/oom.png, проверки выводит в консоль.
"""
import csv
import time
from pathlib import Path

import equations as eq
import numpy as np
import pynvml
import torch
from models import build_model

# точки за пределами сетки по обе стороны от предсказанной границы OOM
OOM_PROBES = ([(512, b) for b in (640, 768, 832, 896, 1024, 1280)]
              + [(384, b) for b in (1280, 1408, 1536, 1664, 2048)]
              + [(256, b) for b in (3072, 3328, 3584)])

BASE_S = [32, 64, 128, 224, 256, 384, 512]
BASE_B = [1, 2, 4, 8, 16, 32, 64, 128, 256]
SEED = 2026
FIELDS = ["S", "B", "latency", "memory", "energy", "is_validation"]


def make_grid(seed=SEED):
    """База + 4 случайных S и 3 случайных B (§4). Точка валидационная, если S или B случайный."""
    rng = np.random.default_rng(seed)
    s_pool = [s for s in range(32, 513, 16) if s not in BASE_S]
    b_pool = [b for b in range(1, 257) if b & (b - 1)]  # не степени двойки
    val_s = sorted(int(v) for v in rng.choice(s_pool, size=4, replace=False))
    val_b = sorted(int(v) for v in rng.choice(b_pool, size=3, replace=False))
    grid = [(s, b, s in val_s or b in val_b)
            for s in sorted(BASE_S + val_s) for b in sorted(BASE_B + val_b)]
    return grid, val_s, val_b


def set_flags():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


@torch.inference_mode()
def measure_latency(model, x, n_warmup=3, n_min=5, n_max=100, target_s=1.0):
    """Медиана времени forward в секундах.

    Синхронизация после каждого прогона, поэтому паузы между запусками кернелов входят во время.
    """
    for _ in range(n_warmup):
        model(x)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    t_begin = time.perf_counter()
    while len(times) < n_max and (len(times) < n_min or time.perf_counter() - t_begin < target_s):
        start.record()
        model(x)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / 1e3)  # мс в с
    return float(np.median(times))


@torch.inference_mode()
def measure_memory(model, x):
    """Пик памяти за forward в байтах, веса и вход включены."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    y = model(x)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del y
    return int(peak)


class EnergyMeter:
    """Счётчик энергии NVML (мДж с момента загрузки драйвера).

    На T4 он обновляется раз в 0.1-0.3 с, поэтому крутим много прогонов
    и выравниваем начало и конец по обновлениям счётчика.
    """

    def __init__(self, index=0):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)

    def read(self):
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)  # мДж

    def wait_update(self, timeout=2.0):
        """Ждёт изменения счётчика, возвращает (время, значение)."""
        v0 = self.read()
        t_limit = time.perf_counter() + timeout
        while True:
            v, t = self.read(), time.perf_counter()
            if v != v0 or t > t_limit:
                return t, v
            time.sleep(0.0005)

    def idle_power(self, seconds=3.0):
        """Мощность простоя в ваттах при активном CUDA-контексте."""
        torch.cuda.synchronize()
        t0, e0 = self.wait_update()
        time.sleep(seconds)
        t1, e1 = self.wait_update()
        return (e1 - e0) / 1e3 / (t1 - t0)


@torch.inference_mode()
def measure_energy(model, x, meter, p_idle, target_s=3.0, k_min=3):
    """Энергия одного forward в джоулях: E(K прогонов) / K.

    Цикл стартует сразу после обновления счётчика. После цикла ждём следующее обновление
    и вычитаем энергию простоя за это время.
    """
    model(x)
    torch.cuda.synchronize()
    t0, e0 = meter.wait_update()
    k = 0
    while k < k_min or time.perf_counter() - t0 < target_s:
        model(x)
        k += 1
    torch.cuda.synchronize()
    t_end = time.perf_counter()
    t1, e1 = meter.wait_update()
    joules = (e1 - e0) / 1e3 - p_idle * (t1 - t_end)
    return joules / k


def run_config(model, s, b, meter, p_idle):
    """Три замера для одной точки, при OOM во всех столбцах "OOM" (§4)."""
    x = None
    try:
        x = torch.randn(b, 3, s, s, device="cuda")
        latency = measure_latency(model, x)
        memory = measure_memory(model, x)
        energy = measure_energy(model, x, meter, p_idle)
        return {"S": s, "B": b, "latency": latency, "memory": memory, "energy": energy}
    except torch.cuda.OutOfMemoryError:
        return {"S": s, "B": b, "latency": "OOM", "memory": "OOM", "energy": "OOM"}
    finally:
        del x
        torch.cuda.empty_cache()


def run_grid(out_csv, seed=SEED, log=print):
    """Замер всей сетки. Строки дописываются по одной, после обрыва запуск продолжится."""
    set_flags()
    torch.manual_seed(0)
    model = build_model().cuda().eval()
    meter = EnergyMeter()
    p_idle = meter.idle_power()
    log(f"idle power: {p_idle:.1f} W")
    grid, _, _ = make_grid(seed)

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_csv.exists():
        with out_csv.open(newline="") as f:
            done = {(int(r["S"]), int(r["B"])) for r in csv.DictReader(f)}
    else:
        with out_csv.open("w", newline="") as f:
            csv.DictWriter(f, FIELDS).writeheader()

    for i, (s, b, is_val) in enumerate(grid, 1):
        if (s, b) in done:
            continue
        t = time.perf_counter()
        row = run_config(model, s, b, meter, p_idle)
        row["is_validation"] = is_val
        with out_csv.open("a", newline="") as f:
            csv.DictWriter(f, FIELDS).writerow(row)
        lat = row["latency"]
        lat_s = "      OOM" if lat == "OOM" else f"{lat * 1e3:9.3f} ms"
        log(f"[{i:3d}/{len(grid)}] S={s:3d} B={b:3d} {'val' if is_val else 'trn'} "
            f"latency {lat_s}  ({time.perf_counter() - t:.1f} s)")
    return p_idle


def oom_probe(configs, out_csv=None):
    """Пик памяти или OOM для точек вне сетки, проверка формулы Memory."""
    set_flags()
    model = build_model().cuda().eval()
    rows = []
    for s, b in configs:
        x = None
        try:
            x = torch.randn(b, 3, s, s, device="cuda")
            rows.append({"S": s, "B": b, "memory": measure_memory(model, x)})
        except torch.cuda.OutOfMemoryError:
            rows.append({"S": s, "B": b, "memory": "OOM"})
        finally:
            del x
            torch.cuda.empty_cache()
    if out_csv is not None:
        with Path(out_csv).open("w", newline="") as f:
            writer = csv.DictWriter(f, ["S", "B", "memory"])
            writer.writeheader()
            writer.writerows(rows)
    return rows


def environment():
    """GPU, версии ПО и память, доступная PyTorch (потолок OOM)."""
    pynvml.nvmlInit()
    driver = pynvml.nvmlSystemGetDriverVersion()
    free_b, total_b = torch.cuda.mem_get_info()
    return {
        "gpu": torch.cuda.get_device_name(0),
        "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
        "driver": driver.decode() if isinstance(driver, bytes) else driver,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "total_bytes": total_b,
        "free_bytes": free_b,  # после создания CUDA-контекста
    }


@torch.inference_mode()
def layer_memory_probe(s, b):
    """Прирост пика памяти на каждом слое и размер его выхода.

    Ожидаем: conv около 1 выхода (+ workspace cuDNN), ReLU 0, MaxPool 3 (выход + индексы int64),
    первый Linear + workspace cuBLAS (выделяется один раз).
    """
    model = build_model().cuda().eval()
    x = torch.randn(b, 3, s, s, device="cuda")
    inp, rows = x, []
    for i, layer in enumerate(model):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        y = layer(x)
        torch.cuda.synchronize()
        out_bytes = y.numel() * y.element_size()
        extra = torch.cuda.max_memory_allocated() - before
        rows.append((i, type(layer).__name__, tuple(y.shape), out_bytes, extra))
        x = y
    del inp, x, y, model
    torch.cuda.empty_cache()
    return rows


def cuda_kernels(s, b):
    """Имена CUDA-кернелов одного forward (torch.profiler)."""
    from torch.profiler import profile, ProfilerActivity
    model = build_model().cuda().eval()
    x = torch.randn(b, 3, s, s, device="cuda")
    with torch.inference_mode():
        model(x)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            model(x)
            torch.cuda.synchronize()
    return [e.name for e in prof.events() if str(e.device_type).endswith("CUDA")]


def plot_oom(probes, capacity, grid_csv, out_png):
    """Граница Memory(S, B) = capacity, пробы и точки сетки."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with Path(grid_csv).open(newline="") as f:
        grid = [(int(r["S"]), int(r["B"])) for r in csv.DictReader(f)]
    s_line = np.linspace(32, 512, 200)
    b_max = (capacity - eq.WEIGHT_BYTES - eq.CUBLAS_WORKSPACE) / (68 * s_line ** 2)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(s_line, b_max, "k-", lw=1.5, label=f"модель: Memory(S, B) = C = {capacity / 2**30:.2f} GiB")
    ax.fill_between(s_line, b_max, 1e5, color="#c0392b", alpha=0.08, label="предсказанная зона OOM")
    ax.scatter(*zip(*grid), s=8, color="grey", label="сетка ДЗ (замеры)")
    fits = [(r["S"], r["B"]) for r in probes if r["memory"] != "OOM"]
    ooms = [(r["S"], r["B"]) for r in probes if r["memory"] == "OOM"]
    if fits:
        ax.scatter(*zip(*fits), s=60, marker="o", facecolors="none", edgecolors="#1e8449", lw=2,
                   label="проба: поместилось")
    if ooms:
        ax.scatter(*zip(*ooms), s=60, marker="x", color="#c0392b", lw=2, label="проба: OOM")
    ax.set_yscale("log", base=2)
    ax.set_ylim(0.8, 8000)
    ax.set_xlabel("image size S, px")
    ax.set_ylabel("batch size B")
    ax.set_title("Граница OOM: модель vs замер")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    here = Path(__file__).resolve().parent
    out_csv = here / "results" / "measurements.csv"
    set_flags()

    env = environment()
    print("== environment ==")
    for k, v in env.items():
        print(f"  {k:12s} {v}")

    # до сетки, чтобы первый Linear показал выделение workspace cuBLAS
    print("\n== per-layer memory: extra peak / output size ==")
    for s, b in [(224, 32), (512, 16)]:
        print(f"  S={s}, B={b}")
        for i, name, shape, out_bytes, extra in layer_memory_probe(s, b):
            print(f"    {i:2d} {name:18s} {str(shape):22s} out {out_bytes:>11,d} B  extra {extra:>11,d} B"
                  f"  x{extra / out_bytes:.2f}")

    print("\n== CUDA kernels per forward (model assumes 17) ==")
    for s, b in [(32, 1), (224, 32), (512, 128)]:
        names = cuda_kernels(s, b)
        print(f"  S={s}, B={b}: {len(names)} kernels")
        for n in dict.fromkeys(names):
            print(f"      {names.count(n):4d} x {n[:100]}")

    print("\n== grid ==")
    run_grid(out_csv)

    print("\n== OOM probes beyond the grid ==")
    probes = oom_probe(OOM_PROBES)
    capacity = env["free_bytes"]
    hits = 0
    for r in probes:
        pred = eq.memory(r["S"], r["B"])
        pred_oom = pred > capacity
        hits += pred_oom == (r["memory"] == "OOM")
        meas = "OOM" if r["memory"] == "OOM" else f"{r['memory'] / 2**30:6.2f} GiB"
        print(f"  S={r['S']:3d} B={r['B']:4d}: measured {meas:>10s} | predicted {pred / 2**30:6.2f} GiB "
              f"-> {'OOM' if pred_oom else 'fits'}")
    print(f"  prediction matches: {hits} / {len(probes)}")
    plot_oom(probes, capacity, out_csv, here / "results" / "figures" / "oom.png")


if __name__ == "__main__":
    main()
