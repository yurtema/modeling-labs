import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import random

# ================= ПАРАМЕТРЫ =================
GRID_SIZE = 100  # размер поля
STEPS = 300  # количество шагов симуляции (без прогрева)
WARMUP = 200  # прогрев
LIGHT_PERIOD = 20  # период переключения светофора
DENSITY_STEP = 0.1  # шаг по плотности
NUM_VERTICAL = 1  # количество вертикальных дорог
NUM_HORIZONTAL = 1  # количество горизонтальных дорог
# =============================================

# направления
UP, DOWN, RIGHT, LEFT = 0, 1, 2, 3

# коды клеток
EMPTY = 0
ROAD = 1
INTERSECTION = 2


def create_grid():
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    # позиции дорог
    vert_pos = np.linspace(GRID_SIZE // (NUM_VERTICAL + 1),
                           GRID_SIZE - GRID_SIZE // (NUM_VERTICAL + 1),
                           NUM_VERTICAL, dtype=int)
    hor_pos = np.linspace(GRID_SIZE // (NUM_HORIZONTAL + 1),
                          GRID_SIZE - GRID_SIZE // (NUM_HORIZONTAL + 1),
                          NUM_HORIZONTAL, dtype=int)

    for x in vert_pos:
        grid[:, x] = ROAD
        grid[:, x - 1] = ROAD  # двусторонка

    for y in hor_pos:
        grid[y, :] = ROAD
        grid[y - 1, :] = ROAD  # двусторонка

    # перекрестки
    for x in vert_pos:
        for y in hor_pos:
            grid[y - 1:y + 1, x - 1:x + 1] = INTERSECTION

    return grid, vert_pos, hor_pos


def populate_cars(grid, density, vert_pos, hor_pos):
    cars = np.zeros_like(grid, dtype=int)
    dirs = np.full_like(grid, -1, dtype=int)

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if grid[i, j] == ROAD and random.random() < density:
                d = -1
                # вертикальные дороги
                for x in vert_pos:
                    if j == x - 1:  # левая полоса вниз
                        d = DOWN
                    elif j == x:  # правая полоса вверх
                        d = UP

                # горизонтальные дороги
                for y in hor_pos:
                    if i == y - 1:  # верхняя полоса влево
                        d = LEFT
                    elif i == y:  # нижняя полоса вправо
                        d = RIGHT

                if d != -1:  # нашли направление
                    cars[i, j] = 1
                    dirs[i, j] = d
    return cars, dirs


def step_sim(grid, cars, dirs, light_state):
    new_cars = np.zeros_like(cars)
    new_dirs = np.full_like(dirs, -1)

    moves = 0

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if cars[i, j] == 1:
                d = dirs[i, j]
                ni, nj = i, j
                if d == UP:
                    ni = (i - 1) % GRID_SIZE
                elif d == DOWN:
                    ni = (i + 1) % GRID_SIZE
                elif d == RIGHT:
                    nj = (j + 1) % GRID_SIZE
                elif d == LEFT:
                    nj = (j - 1) % GRID_SIZE

                allowed = True
                # въезд на перекресток регулируется светофором
                if grid[i, j] != INTERSECTION and grid[ni, nj] == INTERSECTION:
                    if d in (UP, DOWN) and light_state != "VERT":
                        allowed = False
                    if d in (LEFT, RIGHT) and light_state != "HOR":
                        allowed = False

                    # не въезжать в перекресток, если его "выход" заблокирован
                    if allowed:
                        nni, nnj = ni, nj
                        if d == UP:
                            nni = (ni - 1) % GRID_SIZE
                        elif d == DOWN:
                            nni = (ni + 1) % GRID_SIZE
                        elif d == RIGHT:
                            nnj = (nj + 1) % GRID_SIZE
                        elif d == LEFT:
                            nnj = (nj - 1) % GRID_SIZE
                        # если ячейка после перекрестка занята (или уже будет занята), отказываем во въезде
                        if cars[nni, nnj] == 1 or new_cars[nni, nnj] == 1:
                            allowed = False

                if allowed and new_cars[ni, nj] == 0 and cars[ni, nj] == 0:
                    new_cars[ni, nj] = 1
                    new_dirs[ni, nj] = d
                    moves += 1
                else:
                    new_cars[i, j] = 1
                    new_dirs[i, j] = d
    return new_cars, new_dirs, moves


def simulate(density, collect_frames=False):
    grid, vert_pos, hor_pos = create_grid()
    cars, dirs = populate_cars(grid, density, vert_pos, hor_pos)

    total_moves = 0
    total_cars = np.sum(cars)

    frames = []
    for t in range(WARMUP + STEPS):
        light_state = "VERT" if (t // LIGHT_PERIOD) % 2 == 0 else "HOR"
        cars, dirs, moves = step_sim(grid, cars, dirs, light_state)
        if t >= WARMUP:
            total_moves += moves
        if collect_frames and t % 2 == 0:
            frames.append((cars.copy(), light_state))
    if total_cars == 0:
        return 0, frames
    avg_speed = total_moves / (total_cars * STEPS)
    return avg_speed * total_cars, frames


def animate(frames, density, out_gif="sim.gif"):
    fig, ax = plt.subplots()
    im = ax.imshow(frames[0][0], cmap="gray_r", vmin=0, vmax=1)

    def update(frame):
        cars, light_state = frame
        im.set_data(cars)
        ax.set_title(f"Light: {light_state}")
        return [im]

    ani = FuncAnimation(fig, update, frames=frames, blit=True, interval=100)
    ani.save(out_gif, writer=PillowWriter(fps=10))
    plt.close(fig)


if __name__ == "__main__":
    densities = np.arange(0, 1.01, DENSITY_STEP)
    results = []

    frms = 0.5

    for d in densities:
        val, frames = simulate(d, collect_frames=(abs(d - frms) < 1e-6))
        results.append(val)
        print(f"density={d:.1f}, value={val:.2f}")
        if abs(d - frms) < 1e-6:  # гифка
            animate(frames, d, "traffic.gif")

    plt.plot(densities, results, marker="o")
    plt.xlabel("Плотность")
    plt.ylabel("Средняя скорость × число машин")
    plt.title("Эффективность от плотности")
    plt.grid(True)
    plt.savefig("traffic.png")
    plt.show()
