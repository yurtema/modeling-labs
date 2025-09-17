import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
from enum import Enum
from typing import List, Tuple, Optional


rs = np.random.RandomState(seed=1097)

# --- параметры симуляции ---
h, w = 250, 250
time = 500
p_fire = 2e-5
p_grow = 0.02
gif = 0.666
etas = [0.333, gif, 1]



class NeighborhoodType(Enum):
    CROSS = "+"
    DIAGONAL = "x"  # диагонали

class CellState(Enum):
    EMPTY = 0
    FIRING = 1
    TREE = 2


# --- цвета ---
colors = ["#49423D", "orange", "green"]  # пусто, огонь, дерево
cmap_forest = ListedColormap(colors)


# --- генерация начальной сетки ---
def create_ca(w: int, h: int) -> np.ndarray:
    return np.full((h, w), CellState.EMPTY.value, dtype=np.int8)


def init_state(ca: np.ndarray, eta: float, f: int, rs: np.random.RandomState):
    """Инициализация: заполняем eta деревьями, поджигаем f случайных."""
    h, w = ca.shape
    n = h * w
    n_trees = min(n, int(max(0, eta * n)))
    flat_idx = rs.choice(n, size=n_trees, replace=False)
    i_trees, j_trees = flat_idx // w, flat_idx % w
    ca[i_trees, j_trees] = CellState.TREE.value
    if n_trees > 0 and f > 0:
        chosen = rs.choice(n_trees, size=min(f, n_trees), replace=False)
        ca[i_trees[chosen], j_trees[chosen]] = CellState.FIRING.value



def get_cross_neighborhood(cell: Tuple[int, int], ca_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Соседи в крестовой окрестности (4 клетки)"""
    i, j = cell
    h, w = ca_shape
    out = []
    if i > 0:
        out.append((i - 1, j))
    if i < h - 1:
        out.append((i + 1, j))
    if j > 0:
        out.append((i, j - 1))
    if j < w - 1:
        out.append((i, j + 1))
    return out


def get_neuman_neighborhood(cell: Tuple[int, int], shape: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Соседи в фон-Неймановской окрестности (8 клеток)"""
    i, j = cell
    h, w = shape
    neighbors = [
        (i + di, j + dj)
        for di in [-1, 0, 1]
        for dj in [-1, 0, 1]
        if not (di == 0 and dj == 0)
    ]
    return [(ni, nj) for ni, nj in neighbors if 0 <= ni < h and 0 <= nj < w]


# --- обновление 1 клетки ---
def update_cell(ca: np.ndarray, new_ca: np.ndarray,
                cell: Tuple[int, int], neighbors: List[Tuple[int, int]],
                p_fire: float, p_grow: float, rs: np.random.RandomState):
    i, j = cell
    s = ca[i, j]
    if s == CellState.FIRING.value:
        new_ca[i, j] = CellState.EMPTY.value
    elif s == CellState.TREE.value:
        if any(ca[ni, nj] == CellState.FIRING.value for ni, nj in neighbors) or rs.random() < p_fire:
            new_ca[i, j] = CellState.FIRING.value
        else:
            new_ca[i, j] = CellState.TREE.value
    else:  # EMPTY
        new_ca[i, j] = CellState.TREE.value if rs.random() < p_grow else CellState.EMPTY.value


# --- обновление всего поля ---
def update(ca: np.ndarray, nt: NeighborhoodType,
           p_fire: float, p_grow: float, rs: np.random.RandomState) -> np.ndarray:
    h, w = ca.shape
    new_ca = np.empty_like(ca)
    get_neigh = get_cross_neighborhood if nt == NeighborhoodType.CROSS else get_neuman_neighborhood
    for i in range(h):
        for j in range(w):
            neigh = get_neigh((i, j), (h, w))
            update_cell(ca, new_ca, (i, j), neigh, p_fire, p_grow, rs)
    return new_ca


class Statistics:
    def __init__(self):
        self.t, self.a_f, self.a_t, self.a_e = [], [], [], []

    def append(self, t: int, ca: np.ndarray):
        self.t.append(t)
        self.a_f.append(int(np.sum(ca == CellState.FIRING.value)))
        self.a_t.append(int(np.sum(ca == CellState.TREE.value)))
        self.a_e.append(int(np.sum(ca == CellState.EMPTY.value)))

    def summary(self) -> dict:
        """Сводка за все выполнение"""
        arrs = {
            "Firing": np.array(self.a_f, dtype=float),
            "Trees": np.array(self.a_t, dtype=float),
            "Empty": np.array(self.a_e, dtype=float)
        }
        stats = {}
        for k, arr in arrs.items():
            stats[k] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": int(np.min(arr)),
                "max": int(np.max(arr))
            }
        return stats


def plot_three_panels(results, nt: NeighborhoodType, out_png: str):
    rc_pretty()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    colors = {"Trees": "green", "Fire": "orange", "Empty": "#49423D"}

    for ax, (eta, st) in zip(axes, results):
        ax.plot(st.t, st.a_t, lw=2.0, color=colors["Trees"], label="Trees")
        ax.plot(st.t, st.a_f, lw=1.8, color=colors["Fire"], label="Fire")
        ax.plot(st.t, st.a_e, lw=1.5, color=colors["Empty"], label="Empty")

        ax.set_title(f"η = {eta}")
        ax.set_xlabel("t")
        if ax is axes[0]:
            ax.set_ylabel("Counts")
        ax.legend(fontsize=8)

    fig.suptitle(f"Time series — neighborhood {nt.value}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# --- основная функция, запускает симуляцию с заданными параметрами ---
def simulate(ca: np.ndarray, nt: NeighborhoodType, time: int,
             p_fire: float, p_grow: float,
             rs: np.random.RandomState,
             collect_frames: bool = False, frame_stride: int = 5):
    st = Statistics();
    st.append(0, ca)
    frames = [ca.copy()] if collect_frames else []
    for t in range(1, time + 1):
        ca = update(ca, nt, p_fire, p_grow, rs)
        st.append(t, ca)
        if collect_frames and (t % frame_stride == 0 or t == time):
            frames.append(ca.copy())
    return st, frames


# --- графики ---
def rc_pretty():
    plt.rcParams.update({
        "axes.grid": True,
        "grid.alpha": 0.35,
        "axes.titlesize": 18,
        "axes.labelsize": 14,
        "legend.frameon": False,
        "figure.dpi": 140,
        "savefig.dpi": 140
    })



def plot_time_series_single(st: Statistics, nt: NeighborhoodType, out_png: str):
    rc_pretty()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(st.t, st.a_f, lw=2.5, color="orange", label="Горящие деревья $N_f$")
    ax.plot(st.t, st.a_t, lw=2.5, color="green", label="Деревья $N_t$")
    ax.plot(st.t, st.a_e, lw=2.5, color="#3a3a3a", label="Пустые клетки $N_{empty}$")
    ax.set_title(f"Тип окрестности: {nt.value}")
    ax.set_xlabel("Время $t$")
    ax.set_ylabel("Количество $N$")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def save_snapshot(ca: np.ndarray, t: int, out_png: str):
    fig, ax = plt.subplots(figsize=(5.3, 5.0))
    ax.matshow(ca, cmap=cmap_forest)
    ax.set_title(f"Время: {t}")
    ax.set_xlabel("Length");
    ax.set_ylabel("Width");
    ax.set_aspect("equal")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# --- анимация ---
def animate_frames(frames: List[np.ndarray], out_gif: str, eta: float):
    fig, ax = plt.subplots()
    im = ax.matshow(frames[0], cmap=cmap_forest)
    ax.set(xlabel="Length", ylabel="Width")

    def init():
        im.set_data(frames[0])
        ax.set_title(f"Время: 0")
        return [im]

    def animate(k):
        im.set_data(frames[k])
        ax.set_title(f"Время: {k}")
        return [im]

    ani = FuncAnimation(fig, animate, init_func=init,
                        frames=len(frames), interval=80, blit=True)
    try:
        ani.save(out_gif, writer=PillowWriter(fps=12))
        print("GIF saved:", out_gif)
    except Exception as e:
        print("GIF save failed:", e)
    plt.close(fig)


if __name__ == "__main__":
    f = open('results/stats.txt', 'w')
    for nt in [NeighborhoodType.CROSS, NeighborhoodType.DIAGONAL]:
        results = []
        for eta in etas:
            ca = create_ca(w, h)
            init_state(ca, eta, f=3, rs=rs)
            st, frames = simulate(ca, nt, time, p_fire, p_grow, rs,
                                  collect_frames=(eta == gif), frame_stride=2)
            results.append((eta, st))

            f.write(f"\nNeighborhood {nt.value}, η={eta} \n")
            for k, vals in st.summary().items():
                f.write(f"  {k:6s}: mean={vals['mean']:.2f}, std={vals['std']:.2f}, "
                        f"min={vals['min']}, max={vals['max']} \n")
            # статистику в файл
            # гифка только для 0.6
            if eta == gif:
                animate_frames(frames, f"results/forest_eta{eta}_{nt.value}.gif", eta)
        # один общий график на три η
        plot_three_panels(results, nt, f"results/time_series_{nt.value}.png")
    f.close()
