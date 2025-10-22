import numpy as np
import matplotlib.pyplot as plt

# ---------------- БАЗОВЫЕ ПАРАМЕТРЫ ----------------
L = 10_000
ITERATIONS = 1
V_MAX = 5
P_SLOW = 0.5
BURN_STEPS = 10
RECORD_STEPS = 100
SEG_START = 0
SEG_LEN = 1000
RNG = np.random.default_rng()
RHDV_VALUES = np.arange(0.0, 1.01, 0.2)  # от 0 до 1 с шагом 0.2
# ---------------------------------------------------


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
    """Запускает симуляцию и возвращает:
       history_segment — скорости в участке,
       avg_speed — средняя скорость за время записи"""
    n_cars = int(round(density * L))

    positions = RNG.choice(L, size=n_cars, replace=False)
    speeds = np.zeros(n_cars, dtype=int)
    # с вероятностью rhdv — человек, иначе беспилотная
    is_human = RNG.random(n_cars) < rhdv

    history_segment = np.full((RECORD_STEPS, SEG_LEN), -1, dtype=int)
    mean_speeds = np.zeros(RECORD_STEPS, dtype=float)

    for _ in range(BURN_STEPS):
        positions, speeds = step(positions, speeds, is_human)

    for t in range(RECORD_STEPS):
        positions, speeds = step(positions, speeds, is_human)
        mean_speeds[t] = speeds.mean()

        occ = np.full(SEG_LEN, -1, dtype=int)
        rel_pos = (positions - SEG_START) % L
        mask_in_seg = rel_pos < SEG_LEN
        if np.any(mask_in_seg):
            occ[rel_pos[mask_in_seg]] = speeds[mask_in_seg]
        history_segment[t] = occ

    return {
        "positions": positions,
        "speeds": speeds,
        "mean_speeds": mean_speeds,
        "avg_speed": mean_speeds.mean(),
        "history_segment": history_segment,
    }


def visualize_segment(history_segment):
    """Возвращает numpy-изображение (RGB)"""
    h, w = history_segment.shape
    img_data = np.zeros((h, w, 3), dtype=np.uint8)
    colors = {
        -1: (255, 255, 255),
        0: (0, 0, 0),
        1: (255, 0, 0),
        2: (255, 165, 0),
        3: (255, 255, 0),
        4: (0, 255, 0),
    }
    for v in range(-1, V_MAX + 1):
        color = colors.get(v, (0, 200, 0))
        img_data[history_segment == v] = color
    return np.flipud(img_data)


def speed_vs_density():
    densities = np.arange(0.01, 1.0, 0.01)

    plt.figure(figsize=(8, 5))
    for rhdv in RHDV_VALUES:
        avg_speeds = []
        for d in densities:
            temp = []
            for _ in range(ITERATIONS):
                result = run_sim(d, rhdv)
                temp.append(result["avg_speed"])
            avg = np.mean(temp)
            avg_speeds.append(avg * d)
            print(f"density={d:.2f}")
        plt.plot(densities, avg_speeds, label=f"RHDV={rhdv:.1f}")

    plt.xlabel("Плотность автомобилей")
    plt.ylabel("Средняя скорость потока")
    plt.title(f"Скорость потока при разных долях людей (V_MAX={V_MAX}, P_SLOW={P_SLOW})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    res = run_sim(0.2, 0.5)
    img = visualize_segment(res["history_segment"])
    plt.imsave('traffic_segment.png', img)

    speed_vs_density()
