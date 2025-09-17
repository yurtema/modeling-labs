import matplotlib
# принудительно без окон (и чтобы не падал бекэнд в PyCharm)
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

rs = np.random.RandomState(seed=1097)

# -------- ПАРАМЕТРЫ --------
seed = 1097
w, h = 50, 50
f = 1
p_g = 0.02
p_f = 2e-5
sim_time = 50

from enum import Enum
class NeighborhoodType(Enum):
    CROSS = "+"
    NEUMAN = "x"     # диагонали (для этой задачи)

class CellState(Enum):
    EMPTY = 0
    FIRING = 1
    TREE  = 2

from matplotlib.colors import ListedColormap
colors = ["#49423D", "orange", "green"]   # пусто, огонь, дерево
cmap_forest = ListedColormap(colors)

# -------- ВСПОМОГАТЕЛЬНОЕ --------
def create_ca(w: int, h: int):
    return np.full((h, w), CellState.EMPTY.value, dtype=np.int8)

def init_state(ca: np.ndarray, eta: float, f: int):
    h, w = ca.shape
    n = h * w
    n_trees = int(min(n, max(0, int(eta * n))))
    flat_idx = rs.choice(n, size=n_trees, replace=False)
    i_trees = flat_idx // w
    j_trees = flat_idx % w
    ca[i_trees, j_trees] = CellState.TREE.value
    if n_trees > 0 and f > 0:
        k = int(min(f, n_trees))
        choose = rs.choice(n_trees, size=k, replace=False)
        fi = i_trees[choose]; fj = j_trees[choose]
        ca[fi, fj] = CellState.FIRING.value

from typing import List, Tuple
def get_cross_neighborhood(cell: Tuple[int, int], ca_shape: Tuple[int, int]):
    i, j = cell; h, w = ca_shape
    out = []
    if i-1 >= 0: out.append((i-1, j))
    if i+1 <  h: out.append((i+1, j))
    if j-1 >= 0: out.append((i, j-1))
    if j+1 <  w: out.append((i, j+1))
    return out

def get_neuman_neighborhood(cell: Tuple[int, int], ca_shape: Tuple[int, int]):
    i, j = cell; h, w = ca_shape
    cand = [(i-1, j-1), (i-1, j+1), (i+1, j-1), (i+1, j+1)]
    return [(ii, jj) for ii, jj in cand if 0 <= ii < h and 0 <= jj < w]

def update_cell(ca: np.ndarray, new_ca: np.ndarray,
                cell: Tuple[int, int], neigh: List[Tuple[int, int]]):
    i, j = cell; s = ca[i, j]
    if s == CellState.FIRING.value:
        new_ca[i, j] = CellState.EMPTY.value
        return
    if s == CellState.TREE.value:
        for ni, nj in neigh:
            if ca[ni, nj] == CellState.FIRING.value:
                new_ca[i, j] = CellState.FIRING.value
                return
        new_ca[i, j] = CellState.FIRING.value if rs.rand() < p_f else CellState.TREE.value
        return
    # EMPTY
    new_ca[i, j] = CellState.TREE.value if rs.rand() < p_g else CellState.EMPTY.value

def update(ca: np.ndarray, nt: NeighborhoodType):
    h, w = ca.shape
    new_ca = np.copy(ca)
    for i in range(h):
        for j in range(w):
            neigh = (get_cross_neighborhood((i, j), ca.shape)
                     if nt == NeighborhoodType.CROSS
                     else get_neuman_neighborhood((i, j), ca.shape))
            update_cell(ca, new_ca, (i, j), neigh)
    return new_ca

class Statistics:
    def __init__(self):
        self.t, self.a_f, self.a_t, self.a_e = [], [], [], []
    def append(self, t: int, ca: np.ndarray):
        self.t.append(t)
        self.a_f.append(int(np.sum(ca == CellState.FIRING.value)))
        self.a_t.append(int(np.sum(ca == CellState.TREE.value)))
        self.a_e.append(int(np.sum(ca == CellState.EMPTY.value)))

def simulate(ca: np.ndarray, nt: NeighborhoodType, time: int,
             collect_frames: bool=False, frame_stride: int=5):
    st = Statistics(); st.append(0, ca)
    frames = [ca.copy()] if collect_frames else []
    for t in range(1, time+1):
        ca = update(ca, nt)
        st.append(t, ca)
        if collect_frames and (t % frame_stride == 0 or t == time):
            frames.append(ca.copy())
    return st, frames

# -------- ОФОРМЛЕНИЕ, КАК В ПРИМЕРЕ --------
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
    """Один запуск: три кривые N_f, N_t, N_empty с русскими подписями и нужными цветами."""
    rc_pretty()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(st.t, st.a_f,  lw=2.5, color="orange", label="Горящие деревья $N_f$")
    ax.plot(st.t, st.a_t,  lw=2.5, color="green",  label="Деревья $N_t$")
    ax.plot(st.t, st.a_e,  lw=2.5, color="#3a3a3a", label="Пустые клетки $N_{empty}$")
    ax.set_title(f"Тип окрестности: {nt.value}")
    ax.set_xlabel("Время $t$")
    ax.set_ylabel("Количество $N$")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def save_snapshot(ca: np.ndarray, t: int, out_png: str):
    """Картинка решётки с заголовком 'Время: t'."""
    fig, ax = plt.subplots(figsize=(5.3, 5.0))
    im = ax.matshow(ca, cmap=cmap_forest)
    ax.set_title(f"Время: {t}")
    ax.set_xlabel("Length"); ax.set_ylabel("Width"); ax.set_aspect("equal")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

# ================== ЗАПУСКИ ==================
# 1) Вариант-3: сравнение трёх η (как раньше)
etas = [1/3, 2/3, 1.0]
nt = NeighborhoodType.CROSS
T = sim_time
stats_by_eta = {}
rs.seed(seed)

for eta_ in etas:
    ca0 = create_ca(w, h)
    init_state(ca0, eta_, f)
    st, _ = simulate(ca0.copy(), nt, time=T, collect_frames=False)
    stats_by_eta[eta_] = st
# общий график сравнения трёх η
rc_pretty()
fig, axs = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
for eta_, st in stats_by_eta.items():
    lbl = f"η = {eta_:g}"
    axs[0].plot(st.t, st.a_f, label=lbl)
    axs[1].plot(st.t, st.a_t, label=lbl)
    axs[2].plot(st.t, st.a_e, label=lbl)
axs[0].set(title="Горящие деревья", xlabel="t", ylabel="число")
axs[1].set(title="Здоровые деревья", xlabel="t", ylabel="число")
axs[2].set(title="Пустые клетки",   xlabel="t", ylabel="число")
for ax in axs: ax.grid(True); ax.legend()
fig.savefig("variant3_time_series.png", bbox_inches="tight"); plt.close(fig)

# 2) «Как в примере»: один красивый график и снимки поля
eta_demo = 2/3     # можно поменять
rs.seed(seed)
ca_demo = create_ca(w, h)
init_state(ca_demo, eta_demo, f)
st_demo, frames_demo = simulate(ca_demo.copy(), nt, time=T, collect_frames=True, frame_stride=1)

# красивый график
plot_time_series_single(st_demo, nt, out_png="example_like_timeseries.png")
# снимки на t = 7 и 121 (если не хватает кадров — возьмём ближайший)
snap_times = [7, 121]
for t_snap in snap_times:
    t_snap = min(t_snap, len(frames_demo)-1)  # защита от выхода за пределы
    save_snapshot(frames_demo[t_snap], t_snap, out_png=f"snapshot_t{t_snap:03d}.png")

# 3) GIF-анимация для demo-запуска
fig_anim, ax_anim = plt.subplots()
im = ax_anim.matshow(frames_demo[0], cmap=cmap_forest)
ax_anim.set(title=f"Forest fire (eta={eta_demo:g})", xlabel="Length", ylabel="Width")
def init():
    im.set_data(frames_demo[0]); return [im]
def animate(k):
    im.set_data(frames_demo[k]); return [im]
ani = FuncAnimation(fig_anim, animate, init_func=init,
                    frames=len(frames_demo), interval=80, blit=True)
try:
    ani.save(f"variant3_eta{eta_demo:g}.gif", writer=PillowWriter(fps=12))
    print("GIF saved:", f"variant3_eta{eta_demo:g}.gif")
except Exception as e:
    print("GIF save failed:", e)
plt.close(fig_anim)

# 4) Короткая сводка (как у тебя была)
tail = 100
print("=== Сводка по последним", tail, "шагам ===")
for eta_, st in stats_by_eta.items():
    print(f"η={eta_:g}: ⟨burn⟩≈{np.mean(st.a_f[-tail:]):.1f}, "
          f"⟨trees⟩≈{np.mean(st.a_t[-tail:]):.1f}, "
          f"⟨empty⟩≈{np.mean(st.a_e[-tail:]):.1f}")
print("Saved:",
      "variant3_time_series.png,",
      "example_like_timeseries.png,",
      "snapshot_t007.png, snapshot_t121.png,",
      f"variant3_eta{eta_demo:g}.gif")
