import numpy as np
import matplotlib.pyplot as plt

# ---------------- БАЗОВЫЕ ПАРАМЕТРЫ ----------------
L = 1000
ITERATIONS = 5
V_MAX = 3
BURN_STEPS = 10
RECORD_STEPS = 100
SEG_START = 0
SEG_LEN = 10
RNG = np.random.default_rng()

# Параметры модели S-NFS
D = 10  # Расстояние синхронизации
P = 0.1  # Вероятность НЕ случайного торможения

Q = 0.0  # Вероятность медленного старта
R = 0.5  # Вероятность "упреждения" (S_i = 2)



def step(positions, speeds):
    order = np.argsort(positions)
    pos = positions[order]
    v = speeds[order]

    # Вычисляем расстояния до соседних машин
    pos_next = np.roll(pos, -1)
    pos_next2 = np.roll(pos, -2)
    dist1 = (pos_next - pos) % L  # Расстояние до следующей машины
    dist2 = (pos_next2 - pos) % L  # Расстояние до машины через одну
    gap = dist1 - 1  # Свободные клетки до следующей машины

    # Скорости соседних машин (до обновления)
    speeds_next = np.roll(v, -1)

    # ЭТАП 1: Ускорение с условием
    condition = (gap >= D) | (v <= speeds_next)
    v1 = np.where(condition, np.minimum(v + 1, V_MAX), v)

    # ЭТАП 2: Медленный старт
    S_i = np.where(RNG.random(len(pos)) < R, 2, 1)  # Определяем S_i
    dist_S = np.where(S_i == 1, dist1, dist2)  # Выбираем расстояние в зависимости от S_i
    free_gap_S = dist_S - S_i  # Свободные клетки до машины i+S_i

    # Применяем медленный старт с вероятностью Q
    rand_q = RNG.random(len(pos))
    v2 = np.where(rand_q < Q, np.minimum(v1, free_gap_S), v1)

    # ЭТАП 3: Упреждение (всегда)
    v3 = np.minimum(v2, free_gap_S)

    # ЭТАП 4: Случайное торможение
    rand_p = RNG.random(len(pos))
    v4 = np.where(rand_p < (1 - P), np.maximum(v3 - 1, 0), v3)

    # ЭТАП 5: Избежание столкновений
    v4_next = np.roll(v4, -1)  # v4 для машины i+1
    v5 = np.minimum(v4, dist1 - v4_next)  # Безопасная скорость

    # ЭТАП 6: Движение
    pos_new = (pos + v5) % L

    # Возвращаем к исходному порядку
    order_inv = np.empty_like(order)
    order_inv[order] = np.arange(len(order))
    return pos_new[order_inv], v5[order_inv]

def run_sim(density):
    """Запускает симуляцию и возвращает:
       history_segment — скорости в участке,
       avg_speed — средняя скорость за время записи"""
    n_cars = int(round(density * L))

    positions = RNG.choice(L, size=n_cars, replace=False)
    speeds = np.zeros(n_cars, dtype=int)

    history_segment = np.full((RECORD_STEPS, SEG_LEN), -1, dtype=int)
    mean_speeds = np.zeros(RECORD_STEPS, dtype=float)

    # burn-in
    for _ in range(BURN_STEPS):
        positions, speeds = step(positions, speeds)

    for t in range(RECORD_STEPS):
        positions, speeds = step(positions, speeds)
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
        -1: (255, 255, 255),  # пусто
        0: (0, 0, 0),  # черный
        1: (255, 0, 0),  # красный
        2: (255, 165, 0),  # оранжевый
        3: (255, 255, 0),  # желтый
        4: (0, 255, 0),  # зеленый
    }

    for v in range(-1, V_MAX + 1):
        color = colors.get(v, (0, 200, 0))
        img_data[history_segment == v] = color

    img_data = np.flipud(img_data)
    return img_data


def speed_vs_density():
    densities = np.arange(0.01, 1.0, 0.01)
    plt.figure(figsize=(7, 4))

    all_x = []
    all_y = []

    for d in densities:
        for i in range(ITERATIONS):
            result = run_sim(d)
            avg_speed = result["avg_speed"] * d  # поток
            all_x.append(d)
            all_y.append(avg_speed)
            plt.scatter(d, avg_speed, color='blue', s=8, alpha=0.5)

        print(f"density={d:.2f}")

    plt.xlabel("Плотность автомобилей")
    plt.ylabel("Средняя скорость (умноженная на плотность)")
    plt.title(f"Зависимость скорости от плотности (по итерациям, V_MAX={V_MAX})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



# --- пример использования ---
if __name__ == "__main__":
    res = run_sim(0.2)
    print(f"Средняя скорость: {res['avg_speed']:.4f}")

    img = visualize_segment(res["history_segment"])
    # plt.imshow(img)
    plt.imsave('traffic_segment.png', img)

    # график зависимости скорости от плотности
    speed_vs_density()
