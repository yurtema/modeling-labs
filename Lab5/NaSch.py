import numpy as np
import matplotlib.pyplot as plt

# ---------------- БАЗОВЫЕ ПАРАМЕТРЫ ----------------
N_RANDOM_DENSITIES = 300     # количество точек
DENSITY_PRECISION = 4        # точность округления (0.00–1.00)
SMOOTH_WINDOW = 10           # окно сглаживания для линии приближения

L = 1_000
BURN_STEPS = 2000
RECORD_STEPS = 500
V_MAX = 5

P_SLOW = 0.6
RHDV = 0.6
# ---------------------------------------------------

RNG = np.random.default_rng()

def step(positions, speeds, is_human):
    order = np.argsort(positions)
    pos = positions[order]
    v = speeds[order]
    human = is_human[order]

    pos_next = np.roll(pos, -1)
    gap = (pos_next - pos - 1) % L

    v_new = np.minimum(v + 1, V_MAX)
    v_new = np.minimum(v_new, gap)

    # случайные торможения только у людей
    rand = RNG.random(size=v_new.shape)
    slow_mask = (rand < P_SLOW) & human
    v_new[slow_mask] = np.maximum(v_new[slow_mask] - 1, 0)

    pos_new = (pos + v_new) % L

    order_inv = np.empty_like(order)
    order_inv[order] = np.arange(len(order))
    return pos_new[order_inv], v_new[order_inv]


def run_sim(density, rhdv):
    n_cars = int(round(density * L))
    positions = RNG.choice(L, size=n_cars, replace=False)
    speeds = np.zeros(n_cars, dtype=int)
    is_human = RNG.random(n_cars) < rhdv

    mean_speeds = np.zeros(RECORD_STEPS, dtype=float)

    for _ in range(BURN_STEPS):
        positions, speeds = step(positions, speeds, is_human)

    for t in range(RECORD_STEPS):
        positions, speeds = step(positions, speeds, is_human)
        mean_speeds[t] = speeds.mean()

    return mean_speeds.mean()


def smooth_line(x, y, window=10):
    """Скользящее среднее"""
    if len(x) < window:
        return x, y
    x_sorted, y_sorted = zip(*sorted(zip(x, y)))
    x_sorted, y_sorted = np.array(x_sorted), np.array(y_sorted)
    smoothed = np.convolve(y_sorted, np.ones(window)/window, mode='valid')
    # чтобы длины совпадали:
    center_offset = (window - 1) // 2
    x_mid = x_sorted[center_offset: center_offset + len(smoothed)]
    return x_mid, smoothed


def speed_vs_density():
    densities = np.round(RNG.random(N_RANDOM_DENSITIES), DENSITY_PRECISION)
    all_d = []
    all_flow = []

    plt.figure(figsize=(8, 5))
    for d in densities:
        avg_speed = run_sim(d, RHDV)
        flow = avg_speed * d
        plt.scatter(d, flow, color='blue', s=10, alpha=0.6)
        all_d.append(d)
        all_flow.append(flow)

    plt.xlabel("Плотность автомобилей")
    plt.ylabel("Скорость потока")
    plt.title(f"Фундаментальная диаграмма (RHDV={RHDV}, P_SLOW={P_SLOW})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    speed_vs_density()
