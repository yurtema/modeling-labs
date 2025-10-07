import os, math, random, csv
from dataclasses import dataclass, asdict
from typing import List, Optional, Iterable

import matplotlib
matplotlib.use("Agg")  # записываем картинки без GUI
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# --------- утилиты вывода ---------
ROOT = "report_out"
FIG_ID = 1
os.makedirs(ROOT, exist_ok=True)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

def save_fig(fig, path_png: str):
    fig.savefig(path_png, dpi=170, bbox_inches="tight")
    plt.close(fig)

# --------- граф и раскладка ---------
def er_graph(n: int, p: float) -> List[List[int]]:
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                adj[i].append(j); adj[j].append(i)
    return adj
def save_grid(exp_name: str, pos, adj, snapshots: dict, leaders, out_dir: str,
              times=(0, 5, 10, 25, 50, 100)):
    global FIG_ID
    times = [t for t in times if t in snapshots]
    ncols = 3
    nrows = int(math.ceil(len(times) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(6*ncols, 6*nrows))
    axs = axs.ravel()

    for k, t in enumerate(times):
        ax = axs[k]
        draw_graph(pos, adj, snapshots[t], title=f"{exp_name}  t={t}", leaders=leaders, ax=ax)
        ax.text(0.5, -0.08, f"Рис. {FIG_ID}", transform=ax.transAxes,
                ha="center", va="top", fontsize=16)
        FIG_ID += 1

    # убираем пустые ячейки
    for j in range(len(times), len(axs)):
        axs[j].axis("off")

    save_fig(fig, os.path.join(out_dir, f"{exp_name}_grid.png"))


def circle_layout(n: int):
    return [(math.cos(2 * math.pi * k / n), math.sin(2 * math.pi * k / n))
            for k in range(n)]

# --------- шаги модели ---------
def degroot_step(x: List[float], adj: List[List[int]], alpha: float) -> List[float]:
    xn = []
    for i in range(len(x)):
        if not adj[i]:
            xn.append(x[i]); continue
        avg = sum(x[j] for j in adj[i]) / len(adj[i])
        xn.append(alpha * x[i] + (1 - alpha) * avg)
    return xn

def degroot_step_hetero(x: List[float], adj: List[List[int]], alphas: List[float]) -> List[float]:
    xn = []
    for i, a in enumerate(alphas):
        if not adj[i]:
            xn.append(x[i]); continue
        avg = sum(x[j] for j in adj[i]) / len(adj[i])
        xn.append(a * x[i] + (1 - a) * avg)
    return xn

# --------- визуализация ---------
def draw_graph(pos, adj, values, title="", leaders: Optional[Iterable[int]] = None, ax=None):
    """Если ax=None — создаёт фигуру; если ax задан — рисует в него и НЕ закрывает фигуру."""
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        created_fig = True
    else:
        fig = ax.figure

    segs = [[pos[i], pos[j]] for i in range(len(adj)) for j in adj[i] if j > i]
    ax.add_collection(LineCollection(segs, colors="lightgray", linewidths=0.8))
    xs, ys = zip(*pos)
    sc = ax.scatter(xs, ys, c=values, vmin=0, vmax=1, cmap="viridis", s=110, edgecolors="k")
    if leaders:
        lx = [xs[i] for i in leaders]; ly = [ys[i] for i in leaders]
        ax.scatter(lx, ly, s=220, marker="*", facecolors="none", edgecolors="red", linewidths=2.2)
    ax.set_title(title)
    ax.set_aspect("equal"); ax.axis("off")
    plt.colorbar(sc, ax=ax, shrink=0.85)

    return fig


# --------- метрики и протокол ---------
def opinion_range(x: List[float]) -> float:
    return max(x) - min(x)

def linf_residual(x_prev: List[float], x: List[float]) -> float:
    return max(abs(a - b) for a, b in zip(x_prev, x))

@dataclass
class RunSummary:
    exp: str
    n: int
    p: float
    alpha: str
    leaders_pct: float
    T: int
    consensus_eps: float
    reached: bool
    t_consensus: Optional[int]
    range_t0: float
    range_tT: float
    seed: int

def write_timeseries_csv(path_csv: str, rows: List[dict]):
    with open(path_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

def append_summary(path_csv: str, rec: RunSummary):
    file_exists = os.path.exists(path_csv)
    with open(path_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rec).keys()))
        if not file_exists:
            w.writeheader()
        w.writerow(asdict(rec))
def save_grid(exp_name: str, pos, adj, snapshots: dict, leaders, out_dir: str,
              times=(0, 5, 10, 25, 50, 100)):
    """Коллаж 2×3 с подписями 'Рис. N' под каждой панелью."""
    global FIG_ID
    times = [t for t in times if t in snapshots]
    ncols = 3
    nrows = math.ceil(len(times) / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(6*ncols, 6*nrows))
    axs = axs.ravel()

    for k, t in enumerate(times):
        ax = axs[k]
        draw_graph(pos, adj, snapshots[t], title=f"{exp_name}  t={t}", leaders=leaders, ax=ax)
        ax.text(0.5, -0.08, f"Рис. {FIG_ID}", transform=ax.transAxes,
                ha="center", va="top", fontsize=16)
        FIG_ID += 1

    # скрыть пустые ячейки, если их больше, чем times
    for j in range(len(times), len(axs)):
        axs[j].axis("off")

    save_fig(fig, os.path.join(out_dir, f"{exp_name}_grid.png"))

# --------- универсальный прогон ---------
def simulate_and_save(exp_name: str,
                      adj: List[List[int]],
                      x0: List[float],
                      alpha_scalar: Optional[float],
                      alphas_vec: Optional[List[float]],
                      leaders: Optional[Iterable[int]],
                      T: int,
                      pos,
                      out_dir: str,
                      save_times=(0, 5, 10, 25, 50, 100),
                      consensus_eps=1e-3,
                      seed=42,
                      p_val=0.0,
                      grid_mode=False,
                      alpha_desc=""):

    ensure_dir(out_dir)
    x = list(x0)
    r0 = opinion_range(x0)
    t_cons = None

    # --- какие моменты собирать в коллаж(и) ---
    if grid_mode:
        collage1_times = tuple(t for t in (0, 5, 10) if t <= T)
        collage2_times = tuple(t for t in (25, 50, 100) if t <= T)
        collage_times_union = tuple(t for t in (0, 5, 10, 25, 50, 100) if t <= T)
    else:
        collage1_times = tuple(t for t in (0, 50, 100) if t <= T)
        collage2_times = tuple()  # нет второго триптиха
        collage_times_union = collage1_times

    snapshots = {}  # t -> копия вектора мнений

    series = []
    prev = list(x)
    for t in range(0, T + 1):
        # одиночные снимки как раньше
        if t in save_times or t == T:
            fig = draw_graph(pos, adj, x, title=f"{exp_name}  t={t}", leaders=leaders)
            save_fig(fig, os.path.join(out_dir, f"{exp_name}_t{t:03d}.png"))

        # состояния для коллажей
        if t in collage_times_union:
            snapshots[t] = list(x)

        # метрики
        rng = opinion_range(x)
        res = 0.0 if t == 0 else linf_residual(prev, x)
        mean_val = sum(x) / len(x)
        series.append({
            "t": t,
            "range": rng,
            "linf_residual": res,
            "mean": mean_val,
            "var": sum((xi - mean_val) ** 2 for xi in x) / len(x)
        })

        if t_cons is None and rng < consensus_eps:
            t_cons = t

        if t < T:
            prev = x
            if alphas_vec is not None:
                x = degroot_step_hetero(x, adj, alphas_vec)
            else:
                x = degroot_step(x, adj, alpha_scalar)

    write_timeseries_csv(os.path.join(out_dir, f"{exp_name}_timeseries.csv"), series)

    # --- собрать и сохранить коллаж(и) ---
    if snapshots:
        if grid_mode:
            # два отдельных триптиха: 0-5-10 и 25-50-100
            if collage1_times:
                save_triptych(exp_name + " - ",  pos, adj, snapshots, leaders, out_dir,
                              times=collage1_times)
            if collage2_times:
                save_triptych(exp_name + " -- ", pos, adj, snapshots, leaders, out_dir,
                              times=collage2_times)
        else:
            # один стандартный триптих 0-50-100
            save_triptych(exp_name, pos, adj, snapshots, leaders, out_dir,
                          times=collage1_times)

    rec = RunSummary(
        exp=exp_name, n=len(x0), p=p_val, alpha=alpha_desc,
        leaders_pct=0.0 if not leaders else 100.0 * len(list(leaders)) / len(x0),
        T=T, consensus_eps=consensus_eps,
        reached=(t_cons is not None), t_consensus=t_cons,
        range_t0=r0, range_tT=opinion_range(x), seed=seed
    )
    append_summary(os.path.join(ROOT, "summary.csv"), rec)




def save_triptych(exp_name: str, pos, adj, snapshots: dict, leaders, out_dir: str, times=(0, 50, 100)):
    global FIG_ID
    times = [t for t in times if t in snapshots]        # на случай, если T < 100
    fig, axs = plt.subplots(1, len(times), figsize=(6*len(times), 6))
    if len(times) == 1:
        axs = [axs]

    for k, t in enumerate(times):
        ax = axs[k]
        draw_graph(pos, adj, snapshots[t], title=f"{exp_name}  t={t}", leaders=leaders, ax=ax)
        # подпись под панелью: "Рис. N"
        ax.text(0.5, -0.08, f"Рис. {FIG_ID}", transform=ax.transAxes,
                ha="center", va="top", fontsize=22)
        FIG_ID += 1

    save_fig(fig, os.path.join(out_dir, f"{exp_name}_triptych.png"))


# ===================== ПРОГОН ЭКСПЕРИМЕНТОВ =====================
if __name__ == "__main__":
    # общие настройки
    SEED = 1097
    random.seed(SEED)
    N = 50
    T = 100
    pos = circle_layout(N)

    # какие t снимать (подрезаем по T)
    base_snaps = [0, 5, 10, 25, 50, 100]
    save_times = [t for t in base_snaps if t <= T] + ([T] if T not in base_snaps else [])

    # ---------- Эксперимент 1: плотность связей p ----------
    for p in [0.02, 0.1, 0.2]:
        adj = er_graph(N, p)
        x0 = [random.random() for _ in range(N)]
        out_dir = ensure_dir(os.path.join(ROOT, f"Exp1_p={p:g}_alpha=0.5"))
        simulate_and_save(exp_name=f"Exp1_p={p:g}_alpha=0.5",
                          adj=adj, x0=x0,
                          alpha_scalar=0.5, alphas_vec=None,
                          leaders=None, T=T, pos=pos, out_dir=out_dir,
                          save_times=save_times, p_val=p, alpha_desc="0.5", seed=SEED)

    # ---------- Эксперимент 2: упрямство alpha ----------
    p_fixed = 0.1
    adj_fixed = er_graph(N, p_fixed)
    x0_fixed = [random.random() for _ in range(N)]
    for a in [0.0, 0.5, 0.9]:
        out_dir = ensure_dir(os.path.join(ROOT, f"Exp2_p=0.1_alpha={a:g}"))
        simulate_and_save(exp_name=f"Exp2_p=0.1_alpha={a:g}",
                          adj=adj_fixed, x0=x0_fixed,
                          alpha_scalar=a, alphas_vec=None,
                          leaders=None, T=T, pos=pos, out_dir=out_dir,
                          save_times=save_times, p_val=p_fixed, alpha_desc=f"{a:g}", seed=SEED)

    # ---------- Эксперимент 3: двухкластерное упрямство ----------
    p3 = 0.1
    adj3 = er_graph(N, p3)
    x0_3 = [random.random() for _ in range(N)]
    alphas3 = [0.1] * (N // 2) + [0.9] * (N - N // 2)
    out_dir = ensure_dir(os.path.join(ROOT, "Exp3_hetero_alpha_0.1_0.9"))
    simulate_and_save(exp_name="Exp3_hetero_alpha_0.1_0.9",
                      adj=adj3, x0=x0_3,
                      alpha_scalar=None, alphas_vec=alphas3,
                      leaders=None, T=T, pos=pos, out_dir=out_dir,
                      save_times=save_times, p_val=p3, alpha_desc="half:0.1/0.9", seed=SEED, grid_mode=True)

    # ---------- Эксперимент 4: инфлюенсеры (5%), фиксированные мнения ----------
    p4 = 0.1
    adj4 = er_graph(N, p4)
    # стартовые мнения можно сделать «двухполосными» для наглядности
    x0_4 = [0.1] * (N // 2) + [0.9] * (N - N // 2)
    k = max(1, N // 20)  # ~5%
    leaders = set(random.sample(range(N), k))
    a0 = 0.2
    alphas4 = [1.0 if i in leaders else a0 for i in range(N)]  # лидеры не меняют мнение
    out_dir = ensure_dir(os.path.join(ROOT, f"Exp4_leaders={k}_a0={a0:g}"))
    simulate_and_save(exp_name=f"Exp4_leaders={k}_a0={a0:g}",
                      adj=adj4, x0=x0_4,
                      alpha_scalar=None, alphas_vec=alphas4,
                      leaders=leaders, T=T, pos=pos, out_dir=out_dir,
                      save_times=save_times, p_val=p4, alpha_desc=f"a_leaders=1.0, a_others={a0:g}",
                      seed=SEED, grid_mode=True)

    print(f"\nГотово. Смотри папку: {ROOT}")
    print("• Внутри — подпапки Exp1..Exp4 с PNG-снимками и *_timeseries.csv.")
    print("• Общая сводка всех прогонов: report_out/summary.csv")

    # ---------- Эксперимент 5: изолированные вершины ----------
    p5 = 0.1
    adj5 = er_graph(N, p5)
    x0_5 = [random.random() for _ in range(N)]
    # выберем 10% вершин и очистим их списки смежности
    iso_count = N // 10
    isolated = set(random.sample(range(N), iso_count))
    for v in isolated:
        adj5[v] = []
    # также удалим ссылки на эти вершины у соседей
    for u in range(N):
        adj5[u] = [v for v in adj5[u] if v not in isolated]

    out_dir = ensure_dir(os.path.join(ROOT, f"Exp5_isolated={iso_count}"))
    simulate_and_save(exp_name=f"Exp5_isolated={iso_count}",
                      adj=adj5, x0=x0_5,
                      alpha_scalar=0.5, alphas_vec=None,
                      leaders=None, T=T, pos=pos, out_dir=out_dir,
                      save_times=save_times, p_val=p5, alpha_desc="0.5",
                      seed=SEED, grid_mode=True)

