"""Подбор theta и theta_energy по обучающим точкам (§1) и проверка формул на замерах (§8).

Запуск: python calibrate.py
Пишет results/theta.json и results/figures/*.png, ошибки выводит в консоль.
Подбор идёт только по точкам с is_validation == False.
"""
import json
from pathlib import Path

import equations as eq
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, nnls

MATMUL_OPS = [c[0] for c in eq.CONVS] + [lin[0] for lin in eq.LINEARS]
LATENCY_VARIANTS = {
    # одна P на все кернелы + P для FFT conv2 при B > 64, 5 параметров
    "shared_P": ["t0", "t_launch", "bw", "P", "P_conv2_fft"],
    # своя P у каждой Conv и Linear (у cuDNN разные алгоритмы), 12 параметров
    "per_layer_P": ["t0", "t_launch", "bw"] + [f"P_{n}" for n in MATMUL_OPS] + ["P_conv2_fft"],
}
INIT = {"t0": 30e-6, "t_launch": 10e-6, "bw": 200e9, "P": 3e12}


def load_measurements(path):
    """Все строки и строки без OOM с числовыми столбцами."""
    df = pd.read_csv(path)
    df["is_validation"] = df["is_validation"].astype(str).str.lower() == "true"
    ok = df[df["latency"].astype(str) != "OOM"].copy()
    for col in ("latency", "memory", "energy"):
        ok[col] = ok[col].astype(float)
    return df, ok


def _arrays(d, col):
    return d["S"].to_numpy(float), d["B"].to_numpy(float), d[col].to_numpy(float)


def ape(pred, meas):
    """Относительная ошибка по модулю, в долях."""
    return np.abs(np.asarray(pred) - meas) / meas


def theta_from_vector(names, log_values):
    p = {n: float(v) for n, v in zip(names, np.exp(log_values))}
    theta = {"t0": p["t0"], "t_launch": p["t_launch"], "bw": p["bw"], "P_conv2_fft": p["P_conv2_fft"]}
    if "P" in p:
        theta["P"] = p["P"]
    else:
        per = {n: p[f"P_{n}"] for n in MATMUL_OPS}
        theta["P"] = {**per, "default": float(np.median(list(per.values())))}
    return theta


def fit_latency(train, variant="shared_P", n_starts=8, seed=0):
    """МНК по log(T): latency меняется на 4 порядка, поэтому нужна относительная ошибка."""
    names = LATENCY_VARIANTS[variant]
    s, b, t = _arrays(train, "latency")
    x0 = np.log([INIT.get(n, INIT["P"]) for n in names])
    rng = np.random.default_rng(seed)

    def residuals(v):
        return np.log(eq.latency(s, b, theta_from_vector(names, v))) - np.log(t)

    best = None
    for i in range(n_starts):
        start = x0 if i == 0 else x0 + rng.normal(0.0, 1.0, size=len(x0))
        r = least_squares(residuals, start, method="trf")
        if best is None or r.cost < best.cost:
            best = r
    return theta_from_vector(names, best.x)


def cv_latency_error(train, variant, n_starts=3):
    """Кросс-валидация leave-one-S-out на обучающих точках, по ней выбираем вариант."""
    errors = []
    for s_out in sorted(train["S"].unique()):
        theta = fit_latency(train[train["S"] != s_out], variant, n_starts=n_starts)
        s, b, t = _arrays(train[train["S"] == s_out], "latency")
        errors.append(ape(eq.latency(s, b, theta), t))
    return float(np.mean(np.concatenate(errors)))


def fit_energy(train, theta_lat):
    """NNLS для E = P_static*T + e_flop*FLOPs + e_byte*Bytes по относительной ошибке."""
    s, b, e = _arrays(train, "energy")
    a = np.column_stack([eq.latency(s, b, theta_lat), eq.flops(s, b), eq.bytes_moved(s, b)])
    a = a / e[:, None]                      # относительная ошибка
    scale = np.linalg.norm(a, axis=0)       # столбцы различаются на 15 порядков
    coef, _ = nnls(a / scale, np.ones_like(e))
    p_static, e_flop, e_byte = coef / scale
    return {"P_static": float(p_static), "e_flop": float(e_flop), "e_byte": float(e_byte),
            "latency_theta": theta_lat}


def calibrate(csv_path, out_json):
    _, ok = load_measurements(csv_path)
    train = ok[~ok["is_validation"]]
    cv = {v: cv_latency_error(train, v) for v in LATENCY_VARIANTS}
    variant = min(cv, key=cv.get)
    theta = fit_latency(train, variant)
    theta_energy = fit_energy(train, theta)
    result = {
        "latency_variant": variant,
        "latency_cv_mape": cv,
        "theta": theta,
        "theta_energy": theta_energy,
        "candidates": {v: fit_latency(train, v) for v in LATENCY_VARIANTS if v != variant},
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(result, indent=2))
    return result


def predictors(result):
    theta, theta_e = result["theta"], result["theta_energy"]
    return {
        "memory": lambda s, b: eq.memory(s, b),
        "latency": lambda s, b: eq.latency(s, b, theta),
        "energy": lambda s, b: eq.energy(s, b, theta_e),
    }


def counted_flops(s, b):
    """FLOPs сети по FlopCounterMode на meta-устройстве."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    from models import build_model
    with torch.device("meta"):
        model = build_model().eval()
        x = torch.empty(b, 3, s, s)
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        model(x)
    return fc.get_total_flops()


def error_table(ok, result):
    rows = {}
    for col, fn in predictors(result).items():
        row = {}
        for part, d in (("train", ok[~ok["is_validation"]]), ("validation", ok[ok["is_validation"]])):
            s, b, y = _arrays(d, col)
            e = ape(fn(s, b), y)
            row[f"{part} MAPE, %"] = 100 * e.mean()
        row["validation max APE, %"] = 100 * e.max()
        rows[col] = row
    return pd.DataFrame(rows).T.round(1)


def regime_parts(s, b, theta):
    """Latency модели по победившему члену max: launch (+ t0), memory, compute."""
    s, b = np.broadcast_arrays(np.asarray(s, float), np.asarray(b, float))
    parts = {"launch": np.full(s.shape, theta["t0"]), "memory": np.zeros(s.shape), "compute": np.zeros(s.shape)}
    for _, _, tl, tc, tm in eq.latency_terms(s, b, theta):
        t = np.maximum(tl, np.maximum(tc, tm))
        parts["launch"] += np.where((tl >= tc) & (tl >= tm), t, 0)
        parts["compute"] += np.where((tc > tl) & (tc >= tm), t, 0)
        parts["memory"] += np.where((tm > tl) & (tm > tc), t, 0)
    return parts


def regime_errors(ok, theta):
    s, b, t = _arrays(ok, "latency")
    parts = regime_parts(s, b, theta)
    total = sum(parts.values())
    regime = np.select([parts[k] / total > 0.5 for k in ("launch", "compute", "memory")],
                       ["launch", "compute", "memory"], "mixed")
    err = pd.Series(100 * ape(eq.latency(s, b, theta), t), index=ok.index)
    return err.groupby(regime).agg(["size", "mean"]).rename(columns={"size": "points", "mean": "MAPE, %"}).round(1)


def make_figures(ok, result, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    theta = result["theta"]
    pred = predictors(result)
    s_all = sorted(ok["S"].unique())
    col = {s: plt.cm.viridis(i / (len(s_all) - 1)) for i, s in enumerate(s_all)}
    b_line = np.geomspace(1, 256, 200)

    def legend(ax):
        handles = [Line2D([], [], color=col[s], lw=2, label=f"S={s}") for s in s_all]
        handles += [Line2D([], [], color="k", marker="o", ls="", label="замер: train"),
                    Line2D([], [], color="k", marker="o", mfc="none", ls="", label="замер: validation"),
                    Line2D([], [], color="k", lw=1.5, label="модель")]
        ax.legend(handles=handles, fontsize=7, ncol=2)

    def vs_batch(data, y, fn, ylabel, title, name, scale=1.0, per_image=False):
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for s in s_all:
            k = 1 / b_line if per_image else 1
            ax.plot(b_line, fn(s, b_line) * scale * k, color=col[s], lw=1.3)
            d = data[data["S"] == s]
            for is_val, face in ((False, col[s]), (True, "none")):
                dd = d[d["is_validation"] == is_val]
                kk = 1 / dd["B"] if per_image else 1
                ax.scatter(dd["B"], dd[y] * scale * kk, s=22, facecolors=face, edgecolors=col[s], zorder=3)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.set_xlabel("batch size B")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        legend(ax)
        fig.tight_layout()
        fig.savefig(fig_dir / name, dpi=150)
        plt.close(fig)

    # FLOPs: формула против FlopCounterMode, нужен torch
    try:
        fl = ok[["S", "B", "is_validation"]].copy()
        fl["flops"] = [counted_flops(int(s), int(b)) for s, b in zip(fl["S"], fl["B"])]
        err = np.max(np.abs(eq.flops(fl["S"].to_numpy(float), fl["B"].to_numpy(float)) / fl["flops"] - 1))
        print(f"FLOPs: formula vs FlopCounterMode, max relative difference = {err:.2e}")
        vs_batch(fl, "flops", eq.flops, "FLOPs одного forward, GFLOP",
                 "FLOPs(S, B): формула (линии) vs FlopCounterMode (точки)", "flops.png", scale=1e-9)
    except ImportError:
        print("FLOPs check skipped: torch is not installed")

    vs_batch(ok, "memory", pred["memory"], "пиковая память, MiB",
             "Memory(S, B): модель (линии) vs max_memory_allocated (точки)", "memory.png", scale=1 / 2**20)
    vs_batch(ok, "latency", pred["latency"], "latency одного forward, мс",
             "Latency(S, B, θ): модель (линии) vs медиана замеров (точки)", "latency.png", scale=1e3)
    vs_batch(ok, "energy", pred["energy"], "энергия на одно изображение, мДж",
             "Energy(S, B, θ) / B: модель (линии) vs NVML (точки)", "energy.png", scale=1e3, per_image=True)

    tp = ok.assign(throughput=ok["B"] / ok["latency"])
    vs_batch(tp, "throughput", lambda s, b: b / pred["latency"](s, b), "throughput, изображений/с",
             "Throughput = B / Latency: рост в launch-bound, плато в compute-bound", "throughput.png")

    # latency по режимам
    rc = {"launch": "#7f8c8d", "memory": "#2e86c1", "compute": "#c0392b"}
    show = [s for s in (32, 224, 512) if s in s_all]
    fig, axes = plt.subplots(1, len(show), figsize=(15, 4.8), sharey=True)
    for ax, s in zip(np.atleast_1d(axes), show):
        for k, v in regime_parts(s, b_line, theta).items():
            ax.plot(b_line, v * 1e3, color=rc[k], lw=1.5, ls="--", label=f"модель: {k}-часть")
        ax.plot(b_line, pred["latency"](s, b_line) * 1e3, "k-", lw=2, label="модель: сумма")
        d = ok[ok["S"] == s]
        ax.scatter(d["B"], d["latency"] * 1e3, color="k", s=25, zorder=3, label="замер")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.set_title(f"S = {s}")
        ax.set_xlabel("batch size B")
    np.atleast_1d(axes)[0].set_ylabel("время, мс")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle("Из чего складывается latency: launch-, memory- и compute-части модели")
    fig.tight_layout()
    fig.savefig(fig_dir / "regimes.png", dpi=150)
    plt.close(fig)

    # ошибка latency на плоскости (S, B) и границы режимов модели
    s, b, t = _arrays(ok, "latency")
    rel = 100 * (pred["latency"](s, b) / t - 1)
    sg, bg = np.meshgrid(np.linspace(32, 512, 120), np.geomspace(1, 256, 120))
    pg = regime_parts(sg, bg, theta)
    tg = sum(pg.values())
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.contour(sg, bg, pg["launch"] / tg, levels=[0.5], colors=rc["launch"], linewidths=2)
    ax.contour(sg, bg, pg["compute"] / tg, levels=[0.5], colors=rc["compute"], linewidths=2)
    sc = ax.scatter(s, b, c=rel, cmap="RdBu_r", vmin=-40, vmax=40, s=70,
                    edgecolors=np.where(ok["is_validation"], "k", "none"))
    plt.colorbar(sc, label="(модель − замер) / замер, %")
    ax.set_yscale("log", base=2)
    ax.set_xlim(12, 532)
    ax.set_ylim(0.7, 370)
    ax.set_xlabel("image size S, px")
    ax.set_ylabel("batch size B")
    ax.set_title("Ошибка latency по сетке. Линии: границы режимов модели (50 %),\n"
                 "между ними смешанный режим")
    ax.legend([Line2D([], [], color=rc["launch"], lw=2), Line2D([], [], color=rc["compute"], lw=2),
               Line2D([], [], marker="o", mfc="w", mec="k", ls="")],
              ["launch-часть = 50 % (ниже launch-bound)", "compute-часть = 50 % (выше compute-bound)",
               "validation-точка"], fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    fig.tight_layout()
    fig.savefig(fig_dir / "latency_error_map.png", dpi=150)
    plt.close(fig)

    # модель против замеров на всей сетке
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (y, unit, k) in zip(axes, [("memory", "MiB", 1 / 2**20), ("latency", "мс", 1e3), ("energy", "Дж", 1.0)]):
        for is_val, face in ((False, "#2e86c1"), (True, "none")):
            d = ok[ok["is_validation"] == is_val]
            ax.scatter(d[y] * k, pred[y](d["S"].to_numpy(float), d["B"].to_numpy(float)) * k, s=22,
                       facecolors=face, edgecolors="#2e86c1", label="validation" if is_val else "train")
        lim = [ok[y].min() * k * 0.7, ok[y].max() * k * 1.4]
        ax.plot(lim, lim, "k-", lw=1, label="y = x")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.set_xlabel(f"замер, {unit}")
        ax.set_ylabel(f"модель, {unit}")
        ax.set_title(y)
        ax.legend(fontsize=8)
    fig.suptitle("Модель против замеров на всей сетке")
    fig.tight_layout()
    fig.savefig(fig_dir / "parity.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    res = calibrate(here / "results" / "measurements.csv", here / "results" / "theta.json")
    print(json.dumps({k: res[k] for k in ("latency_variant", "latency_cv_mape", "theta")}, indent=2))
    te = res["theta_energy"]
    print(f"theta_energy: P_static = {te['P_static']:.1f} W, e_flop = {te['e_flop'] * 1e12:.2f} pJ/FLOP, "
          f"e_byte = {te['e_byte'] * 1e12:.1f} pJ/B")

    _, ok = load_measurements(here / "results" / "measurements.csv")
    print("\nErrors (train = base grid, validation = random S or B):")
    print(error_table(ok, res).to_string())
    print("\nLatency error by predicted regime:")
    print(regime_errors(ok, res["theta"]).to_string())
    make_figures(ok, res, here / "results" / "figures")
    print(f"\nfigures -> {here / 'results' / 'figures'}")
