from typing import List, Tuple, Optional
from enum import Enum

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation


# =========================
# Конфигурация
# =========================

rs = np.random.RandomState(seed=1097)

class NeighborhoodType(Enum):
    CROSS = "+"
    NEUMAN = "x"


class CellState(Enum):
    EMPTY = 0
    FIRING = 1
    TREE = 2


# Цвета: пусто, огонь, дерево
COLORS = ["#49423D", "orange", "green"]
cmap_forest = ListedColormap(COLORS)


# =========================
# Вспомогательные функции
# =========================

def create_ca(h: int, w: int) -> np.ndarray:
    """Создать пустой клеточный автомат (h×w)."""
    return np.zeros((h, w), dtype=int)


def init_state(ca: np.ndarray, tree_density: float, fire_count: int, rng: np.random.RandomState):
    """Инициализировать состояние: деревья + случайные пожары."""
    h, w = ca.shape

    tree_mask = rng.random(ca.shape) < tree_density
    ca[tree_mask] = CellState.TREE.value

    tree_indices = np.argwhere(tree_mask)
    if len(tree_indices) > 0:
        burning = rng.choice(len(tree_indices), size=min(fire_count, len(tree_indices)), replace=False)
        for idx in burning:
            i, j = tree_indices[idx]
            ca[i, j] = CellState.FIRING.value


def get_cross_neighborhood(cell: Tuple[int, int], shape: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Соседи в '+'-окрестности."""
    i, j = cell
    h, w = shape
    candidates = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
    return [(ni, nj) for ni, nj in candidates if 0 <= ni < h and 0 <= nj < w]


def get_neuman_neighborhood(cell: Tuple[int, int], shape: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Соседи в фон-Неймановской окрестности (8 направлений)."""
    i, j = cell
    h, w = shape
    neighbors = [
        (i + di, j + dj)
        for di in [-1, 0, 1]
        for dj in [-1, 0, 1]
        if not (di == 0 and dj == 0)
    ]
    return [(ni, nj) for ni, nj in neighbors if 0 <= ni < h and 0 <= nj < w]


def update_cell(
    ca: np.ndarray, new_ca: np.ndarray,
    cell: Tuple[int, int], neighbors: List[Tuple[int, int]],
    p_fire: float, p_grow: float, rng: np.random.RandomState,
):
    """Обновить одну клетку."""
    i, j = cell
    state = ca[i, j]

    has_fire = any(ca[ni, nj] == CellState.FIRING.value for ni, nj in neighbors)

    if state == CellState.FIRING.value:
        new_ca[i, j] = CellState.EMPTY.value
    elif state == CellState.TREE.value:
        if has_fire or rng.random() < p_fire:
            new_ca[i, j] = CellState.FIRING.value
        else:
            new_ca[i, j] = CellState.TREE.value
    else:  # EMPTY
        if rng.random() < p_grow:
            new_ca[i, j] = CellState.TREE.value
        else:
            new_ca[i, j] = CellState.EMPTY.value


def update(
    ca: np.ndarray, nt: NeighborhoodType, p_fire: float,
    p_grow: float, rng: np.random.RandomState) -> np.ndarray:
    """Обновить весь автомат на один шаг."""
    h, w = ca.shape
    new_ca = np.empty_like(ca)

    get_neighbors = (
        get_cross_neighborhood if nt == NeighborhoodType.CROSS else get_neuman_neighborhood
    )

    for i in range(h):
        for j in range(w):
            neighbors = get_neighbors((i, j), (h, w))
            update_cell(ca, new_ca, (i, j), neighbors, p_fire, p_grow, rng)

    return new_ca


def animate_ca(
    ca: np.ndarray,
    nt: NeighborhoodType,
    frames: int = 100,
    p_fire: float = 0.001,
    p_grow: float = 0.01,
    interval: int = 100,
    rng: Optional[np.random.RandomState] = None,
):
    """Создать анимацию."""
    if rng is None:
        rng = np.random.RandomState()

    fig, ax = plt.subplots(figsize=(8, 8))
    img = ax.imshow(ca, cmap=cmap_forest, vmin=0, vmax=2, interpolation="none")
    ax.axis("off")

    def update_frame(frame):
        nonlocal ca
        ca = update(ca, nt, p_fire, p_grow, rng)
        img.set_array(ca)

        trees = np.sum(ca == CellState.TREE.value)
        fires = np.sum(ca == CellState.FIRING.value)
        empty = np.sum(ca == CellState.EMPTY.value)
        ax.set_title(f"Шаг {frame} | Деревья={trees}, Огонь={fires}, Пусто={empty}")

        return [img]

    ani = FuncAnimation(fig, update_frame, frames=frames, interval=interval, blit=True)
    plt.show()
    return ani


# =========================
# Пример запуска
# =========================

if __name__ == "__main__":
    size = 20

    ca1 = create_ca(size, size)
    init_state(ca1, tree_density=0.5, fire_count=1, rng=rs)

    ca2 = create_ca(size, size)
    init_state(ca2, tree_density=0.67, fire_count=1, rng=rs)

    ca3 = create_ca(size, size)
    init_state(ca3, tree_density=1.0, fire_count=1, rng=rs)

    animate_ca(ca1, NeighborhoodType.NEUMAN, frames=500, p_fire=0.00002, p_grow=0.02, interval=0, rng=rs)
    animate_ca(ca2, NeighborhoodType.CROSS,  frames=500, p_fire=0.00002, p_grow=0.02, interval=0, rng=rs)
    animate_ca(ca3, NeighborhoodType.CROSS,  frames=500, p_fire=0.00002, p_grow=0.02, interval=0, rng=rs)
