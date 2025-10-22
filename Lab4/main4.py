import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

# ---------------- БАЗОВЫЕ ПАРАМЕТРЫ ----------------
L = 500
N_LANES = 100
ITERATIONS = 1
V_MAX = 5
P_SLOW = 0.5
P_LC = 1.0
BURN_STEPS = 100
RECORD_STEPS = 300
SEG_START = 0
SEG_LEN = 500
LOOP = False  # если True — верх и низ соединены по вертикали
RNG = np.random.default_rng()

colors = ['white', 'black', 'darkred', 'red', 'orange', 'yellow', 'green']
cmap = LinearSegmentedColormap.from_list('traffic', colors, N=len(colors))


# ---------------------------------------------------

def compute_gaps(positions, L):
    if len(positions) == 0:
        return np.array([], dtype=int)
    sorted_indices = np.argsort(positions)
    sorted_pos = positions[sorted_indices]
    gaps = np.zeros_like(sorted_pos)
    for i in range(len(sorted_pos)):
        nxt = sorted_pos[(i + 1) % len(sorted_pos)]
        gaps[i] = (nxt - sorted_pos[i] - 1) % L
    gaps_original = np.zeros_like(gaps)
    gaps_original[sorted_indices] = gaps
    return gaps_original


def step_single_lane(positions, speeds):
    if len(positions) == 0:
        return positions.copy(), speeds.copy()
    order = np.argsort(positions)
    pos = positions[order]
    v = speeds[order].astype(int)

    pos_next = np.roll(pos, -1)
    gap = (pos_next - pos - 1) % L

    v_new = np.minimum(v + 1, V_MAX)
    v_new = np.minimum(v_new, gap)
    rand = RNG.random(size=v_new.shape)
    slow_mask = rand < P_SLOW
    v_new[slow_mask] = np.maximum(v_new[slow_mask] - 1, 0)

    pos_new = (pos + v_new) % L

    order_inv = np.empty_like(order)
    order_inv[order] = np.arange(len(order))
    return pos_new[order_inv], v_new[order_inv]


def get_neighbor_cars(positions, target_pos, L):
    n = len(positions)
    if n == 0:
        return None, None, L, L

    sorted_pos = np.sort(positions)
    idx = np.searchsorted(sorted_pos, target_pos, side='right')
    front = sorted_pos[idx % n]
    back = sorted_pos[(idx - 1) % n]
    gap_front = (front - target_pos - 1) % L
    gap_back = (target_pos - back - 1) % L
    return front, back, gap_front, gap_back


def lane_change(positions_lanes, speeds_lanes):
    new_positions = [np.copy(pos) for pos in positions_lanes]
    new_speeds = [np.copy(spd) for spd in speeds_lanes]

    for lane in range(N_LANES):
        if len(new_positions[lane]) == 0:
            continue

        for i in range(len(new_positions[lane]) - 1, -1, -1):
            pos = new_positions[lane][i]
            v = int(new_speeds[lane][i])

            _, _, gap_front_cur, gap_back_cur = get_neighbor_cars(new_positions[lane], pos, L)

            # --- лево (lane - 1) ---
            left = lane - 1
            if left < 0 and LOOP:
                left = N_LANES - 1
            if 0 <= left < N_LANES and (left != lane):
                front_left, back_left, gap_front_left, gap_back_left = get_neighbor_cars(
                    new_positions[left], pos, L)
                stim_left = (gap_front_cur < v) and (gap_front_left > gap_front_cur)
                safe_left = gap_back_left >= 1
                if stim_left and safe_left and RNG.random() < P_LC:
                    new_positions[left] = np.append(new_positions[left], pos)
                    new_speeds[left] = np.append(new_speeds[left], v)
                    new_positions[lane] = np.delete(new_positions[lane], i)
                    new_speeds[lane] = np.delete(new_speeds[lane], i)
                    continue

            # --- право (lane + 1) ---
            right = lane + 1
            if right >= N_LANES and LOOP:
                right = 0
            if 0 <= right < N_LANES and (right != lane):
                front_right, back_right, gap_front_right, gap_back_right = get_neighbor_cars(
                    new_positions[right], pos, L)
                stim_right = (gap_front_cur < v) and (gap_front_right > gap_front_cur)
                safe_right = gap_back_right >= 1
                if stim_right and safe_right and RNG.random() < P_LC:
                    new_positions[right] = np.append(new_positions[right], pos)
                    new_speeds[right] = np.append(new_speeds[right], v)
                    new_positions[lane] = np.delete(new_positions[lane], i)
                    new_speeds[lane] = np.delete(new_speeds[lane], i)
                    continue

    return new_positions, new_speeds


def step_multi_lane(positions_lanes, speeds_lanes):
    new_positions = []
    new_speeds = []
    for lane in range(N_LANES):
        if len(positions_lanes[lane]) > 0:
            pos_new, spd_new = step_single_lane(positions_lanes[lane], speeds_lanes[lane])
            new_positions.append(pos_new)
            new_speeds.append(spd_new)
        else:
            new_positions.append(np.array([], dtype=int))
            new_speeds.append(np.array([], dtype=int))
    new_positions, new_speeds = lane_change(new_positions, new_speeds)
    return new_positions, new_speeds


def run_sim_multi(density):
    n_cars_total = int(round(density * L * N_LANES))
    positions_lanes = [np.array([], dtype=int) for _ in range(N_LANES)]
    speeds_lanes = [np.array([], dtype=int) for _ in range(N_LANES)]

    for _ in range(n_cars_total):
        lane = int(RNG.integers(0, N_LANES))
        pos = int(RNG.integers(0, L))
        positions_lanes[lane] = np.append(positions_lanes[lane], pos)
        speeds_lanes[lane] = np.append(speeds_lanes[lane], 0)

    history_segment = np.full((RECORD_STEPS, N_LANES, SEG_LEN), -1, dtype=int)
    mean_speeds = np.zeros(RECORD_STEPS, dtype=float)

    for _ in range(BURN_STEPS):
        positions_lanes, speeds_lanes = step_multi_lane(positions_lanes, speeds_lanes)

    for t in range(RECORD_STEPS):
        positions_lanes, speeds_lanes = step_multi_lane(positions_lanes, speeds_lanes)
        all_speeds = np.concatenate([s for s in speeds_lanes]) if any(len(s) for s in speeds_lanes) else np.array([], dtype=int)
        mean_speeds[t] = np.mean(all_speeds) if len(all_speeds) > 0 else 0.0

        for lane in range(N_LANES):
            occ = np.full(SEG_LEN, -1, dtype=int)
            if len(positions_lanes[lane]) > 0:
                rel_pos = (positions_lanes[lane] - SEG_START) % L
                mask_in_seg = rel_pos < SEG_LEN
                if np.any(mask_in_seg):
                    spds = speeds_lanes[lane][mask_in_seg]
                    occ[rel_pos[mask_in_seg]] = np.where(spds == 0, 0, spds)
            history_segment[t, lane] = occ

    return {
        "positions_lanes": positions_lanes,
        "speeds_lanes": speeds_lanes,
        "mean_speeds": mean_speeds,
        "avg_speed": mean_speeds.mean(),
        "history_segment": history_segment,
    }


def create_animation(history_segment, filename='traffic_multi_lane.gif'):
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
    data = history_segment

    im = ax.imshow(
        data[0],
        cmap=cmap,
        vmin=-1,
        vmax=V_MAX,
        interpolation='nearest',
        aspect='equal'
    )

    ax.set_facecolor('white')
    ax.set_xlabel('Позиция на дороге')
    ax.set_ylabel('Полоса (lane 0 = верх)')
    ax.set_title('Многополосное движение', color='black')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Скорость', color='black')
    cbar.ax.yaxis.set_tick_params(color='black')
    cbar.outline.set_edgecolor('black')
    cbar.set_ticks(list(range(-1, V_MAX + 1)))
    cbar.set_ticklabels(['Пусто'] + [str(i) for i in range(0, V_MAX + 1)])
    ax.grid(False)
    fig.patch.set_facecolor('white')

    def update(frame):
        im.set_data(data[frame])
        ax.set_title(f'Шаг {frame}', color='black')
        return [im]

    anim = animation.FuncAnimation(fig, update, frames=data.shape[0], interval=50, blit=True)
    anim.save(filename, writer='pillow', fps=10)
    plt.close()
    print(f"Анимация сохранена как {filename}")


if __name__ == "__main__":
    print("Запуск многополосной симуляции...")
    res = run_sim_multi(0.2)
    print(f"Средняя скорость: {res['avg_speed']:.4f}")
    print("Создание анимации...")
    create_animation(res["history_segment"])
