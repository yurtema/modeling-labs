import random
from typing import Dict, List, Optional, Tuple
from numpy.random import Generator, default_rng
import numpy as np
import simpy as sim
import math
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# ----------------- ФИЗИКА / ПАРАМЕТРЫ (константы) -----------------
g = np.array([0.0, 0.0, -9.81])
H = 2500.0  # Высота вулкана
Dc = 200.0  # Глубина кратера от вершины
Rc = 30.0  # Радиус кратера

Z_GROUND = 0

rho = 2600.0
k_p = 0.6


mu_v, sigma_v = 90.0, 12.0
mu_th, sigma_th = np.deg2rad(2.0), np.deg2rad(2.0)
R_min, R_max = 0.5, 1.2
beta_erup = 0.6
n_per_erup = 30
seed = random.randint(0, 1000)


# ----------------- УТИЛИТЫ / МАССА -----------------
def mass_from_radius(R: float, rho_local: float = rho) -> float:
    return rho_local * (4.0 / 3.0) * np.pi * R ** 3


class Bomb:
    __slots__ = ("t_erup", "t0", "r", "v", "m", "R", "_t_ground", "_collided", "_landed")

    def __init__(self, t_erup: float, r0: np.ndarray, v0: np.ndarray, mass: float, radius: float):
        self.t_erup = float(t_erup)
        self.t0 = float(t_erup)
        self.r = r0.astype(float)
        self.v = v0.astype(float)
        self.m = float(mass)
        self.R = float(radius)
        self._t_ground: Optional[float] = None
        self._collided = False
        self._landed = False

    def calc_r(self, t: float) -> np.ndarray:
        dt = t - self.t0
        return self.r + self.v * dt + 0.5 * g * dt * dt

    def calc_v(self, t: float) -> np.ndarray:
        return self.v + g * (t - self.t0)

    def is_collided(self) -> bool:
        return self._collided


# ----------------- STATE (собирает коллекции, чтобы избежать глобов) -----------------
class SimulationState:
    def __init__(self):
        self.flyings: List[Bomb] = []
        self.fallens: List[Bomb] = []
        self.processes: Dict[Bomb, List[sim.Process]] = {}


# ----------------- Сэмплинг начальных условий -----------------
def sample_initial(rs: Generator, Rc: float,
                   mu_v: float, sigma_v: float,
                   mu_th: float, sigma_th: float,
                   R_min: float, R_max: float) -> Tuple[np.ndarray, np.ndarray, float, float]:
    # равномерно по площади кратера
    d = np.sqrt(rs.uniform(0.0, 1.0)) * Rc
    alpha = rs.uniform(0.0, 2 * np.pi)
    z0 = rs.uniform(H - Dc, H)
    r0 = np.array([d * np.cos(alpha), d * np.sin(alpha), z0], float)

    v_mod = max(0.0, rs.normal(mu_v, sigma_v))
    th = float(np.clip(rs.normal(mu_th, sigma_th), 0.0, np.pi / 2.0))
    phi = rs.uniform(0.0, 2 * np.pi)

    v0 = np.array([
        v_mod * np.sin(th) * np.cos(phi),
        v_mod * np.sin(th) * np.sin(phi),
        v_mod * np.cos(th)
    ], float)

    R = rs.uniform(R_min, R_max)
    m = mass_from_radius(R)
    return r0, v0, m, R


# ----------------- ПОМОЩНИКИ ВРЕМЕНИ -----------------
def when_ground(b: Bomb):
    """Время достижения локальной земли Z_GROUND (абсолютное время)."""
    z0, vz, gz = b.r[2], b.v[2], g[2]
    a, bb, c = 0.5 * gz, vz, (z0 - Z_GROUND)
    if abs(a) < 1e-14:
        if abs(bb) < 1e-14:
            return None
        dt = -c / bb
        return b.t0 + dt if dt > 1e-9 else None
    D = bb * bb - 4 * a * c
    if D < 0:
        return None
    sD = math.sqrt(D)
    dt1 = (-bb - sD) / (2 * a)
    dt2 = (-bb + sD) / (2 * a)
    dts = [dt for dt in (dt1, dt2) if dt >= 0]
    return b.t0 + min(dts) if dts else None


def when_collision(b1: Bomb, b2: Bomb):
    """
    Решаем квадратичное уравнение для расстояния между центрами = R1+R2.
    Возвращает абсолютное время столкновения, если оно произойдёт ДО падения объектов.
    """
    now = max(b1.t0, b2.t0)
    Rsum = b1.R + b2.R
    r = b1.calc_r(now) - b2.calc_r(now)
    v = b1.calc_v(now) - b2.calc_v(now)
    A = float(np.dot(v, v))
    B = float(2 * np.dot(r, v))
    C = float(np.dot(r, r) - Rsum * Rsum)
    if C <= 0.0 and B >= 0.0:  # уже касаются и расходятся
        return None
    if A < 1e-14:
        return None
    D = B * B - 4 * A * C
    if D < 0:
        return None
    sD = math.sqrt(D)
    taus = [(-B - sD) / (2 * A), (-B + sD) / (2 * A)]
    taus = [tau for tau in taus if tau > 1e-9]
    if not taus:
        return None
    tcol = now + min(taus)
    tg1 = b1._t_ground or when_ground(b1)
    tg2 = b2._t_ground or when_ground(b2)
    if tg1 is None or tg2 is None:
        return None
    return tcol if (tcol < tg1 - 1e-9 and tcol < tg2 - 1e-9) else None


def calc_collision(t: float, b1: Bomb, b2: Bomb):
    """Импульсное столкновение с коэффициентом восстановления k_p."""
    r1, v1 = b1.calc_r(t), b1.calc_v(t)
    r2, v2 = b2.calc_r(t), b2.calc_v(t)
    n = r1 - r2
    L = np.linalg.norm(n)
    if L < 1e-9:
        return v1, v2  # слишком близко, не считаем столкновение
    n /= L

    rel_vn = np.dot(v1 - v2, n)
    if rel_vn >= 0.0:
        return v1, v2  # расходятся, не сталкиваются

    m1, m2 = b1.m, b2.m
    J = -(1 + k_p) * rel_vn / (1 / m1 + 1 / m2)
    v1p = v1 + (J / m1) * n
    v2p = v2 - (J / m2) * n
    return v1p, v2p


# ----------------- ПРОЦЕССЫ SIMPY (переработаны для передачи state) -----------------

def clear_queue(state: SimulationState, b: Bomb):
    if b not in state.processes:
        return
    for p in state.processes[b]:
        if not p.triggered:
            try:
                p.interrupt()
            except RuntimeError:
                # уже завершён/неактивен
                pass
    state.processes[b].clear()


def gen_bombs(env: sim.Environment, n: int, rs: Generator) -> List[Bomb]:
    bombs = []
    for _ in range(n):
        r0, v0, m, R = sample_initial(rs, Rc, mu_v, sigma_v, mu_th, sigma_th, R_min, R_max)
        bombs.append(Bomb(env.now, r0, v0, m, R))
    return bombs


def ground(env: sim.Environment, dt: float, b: Bomb, state: SimulationState):
    try:
        yield env.timeout(max(0.0, dt))
    except sim.Interrupt:
        return
    clear_queue(state, b)
    b._landed = True
    b.r = b.calc_r(env.now)
    b.r[2] = Z_GROUND  # точно фиксируем касание земли
    b.v = b.calc_v(env.now)
    b.t0 = env.now
    if b in state.flyings:
        state.flyings.remove(b)
    state.fallens.append(b)


def collision(env: sim.Environment, dt: float, b1: Bomb, b2: Bomb, state: SimulationState):
    try:
        yield env.timeout(max(0.0, dt))
    except sim.Interrupt:
        return
    if (b1 not in state.flyings) or (b2 not in state.flyings):
        return
    t = env.now
    v1p, v2p = calc_collision(t, b1, b2)
    r1c, r2c = b1.calc_r(t), b2.calc_r(t)
    b1.r, b1.v, b1.t0 = r1c, v1p, t
    b2.r, b2.v, b2.t0 = r2c, v2p, t
    b1._collided = True
    b2._collided = True
    clear_queue(state, b1)
    clear_queue(state, b2)

    b1._t_ground = when_ground(b1)
    b2._t_ground = when_ground(b2)
    if b1._t_ground is not None:
        state.processes.setdefault(b1, []).append(env.process(ground(env, b1._t_ground - env.now, b1, state)))
    if b2._t_ground is not None:
        state.processes.setdefault(b2, []).append(env.process(ground(env, b2._t_ground - env.now, b2, state)))

    # новые коллизии с остальными
    for a, b in ((b1, b2), (b2, b1)):
        for other in list(state.flyings):
            if other is a or other is b:
                continue
            tnext = when_collision(a, other)
            if tnext is not None:
                p = env.process(collision(env, tnext - env.now, a, other, state))
                state.processes.setdefault(a, []).append(p)
                state.processes.setdefault(other, []).append(p)


def eruption_process(env: sim.Environment, rs: Generator, state: SimulationState,
                     allowed_collisions: bool, total_bombs: int):
    """Процесс периодических извержений до тех пор, пока не будет создано total_bombs штук."""
    created = 0
    while created < total_bombs:
        dt = rs.exponential(beta_erup)
        yield env.timeout(dt)

        # Сколько осталось до лимита
        n_to_create = min(n_per_erup, total_bombs - created)
        new_bombs = gen_bombs(env, n_to_create, rs)
        created += n_to_create

        for b in new_bombs:
            state.flyings.append(b)
            state.processes.setdefault(b, [])
            b._t_ground = when_ground(b)
            if b._t_ground is not None:
                state.processes[b].append(env.process(ground(env, b._t_ground - env.now, b, state)))
            if not allowed_collisions:
                continue
            for other in list(state.flyings):
                if other is b:
                    continue
                tcol = when_collision(b, other)
                if tcol is not None:
                    p = env.process(collision(env, tcol - env.now, b, other, state))
                    state.processes[b].append(p)
                    state.processes.setdefault(other, []).append(p)
    # после того как все созданы — просто выходим, не порождая больше


def simulate(total_bombs: int, allowed_collisions: bool, seed: int = seed) -> SimulationState:
    """Запуск симуляции до тех пор, пока все total_bombs не упадут."""
    env = sim.Environment()
    rs = default_rng(seed)
    state = SimulationState()

    # запускаем извержения
    env.process(eruption_process(env, rs, state, allowed_collisions, total_bombs))

    env.run()

    return state


def draw_percentile_circles(ax, xy: np.ndarray, ps=(0.75, 0.99), ls='-'):
    if len(xy) == 0:
        return
    r = np.linalg.norm(xy, axis=1)
    for p in ps:
        rp = float(np.quantile(r, p))
        ax.add_patch(plt.Circle((0, 0), rp, fill=False, ls=ls, lw=1.4, zorder=10))
        rlab = rp / math.sqrt(2)
        ax.text(rlab, -rlab, f"{int(p * 100)}%",
                ha='center', va='center', color='white', zorder=11,
                path_effects=[pe.withStroke(linewidth=2, foreground='black')])
    return float(np.quantile(r, ps[-1]))  # возвращаем радиус 99%

def save_static_topview(xy_nc: np.ndarray, xy_not: np.ndarray, xy_col: np.ndarray, xy_all: np.ndarray, fname: str = "vulkano_topview.png"):
    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    ax.set_aspect('equal')

    # Жерло
    ax.add_patch(plt.Circle((0, 0), Rc, fill=True, ec='k', fc='tab:orange', alpha=0.8, lw=1.0, zorder=0))

    # Бомбы
    if len(xy_nc):
        ax.scatter(xy_nc[:, 0], xy_nc[:, 1], s=18, facecolors='none', edgecolors='green', marker='s',
                   label='No Collision Allowed')
    if len(xy_not):
        ax.scatter(xy_not[:, 0], xy_not[:, 1], s=18, marker='o', label='Not Collided (Collision Allowed)')
    if len(xy_col):
        ax.scatter(xy_col[:, 0], xy_col[:, 1], s=24, marker='^', label='Collided (Collision Allowed)', c='red')

    # Перцентильные круги
    max_r = 0
    if len(xy_nc):
        r_nc = draw_percentile_circles(ax, xy_nc, ps=(0.75, 0.99), ls='--')
        max_r = max(max_r, r_nc)
    if len(xy_all):
        r_all = draw_percentile_circles(ax, xy_all, ps=(0.75, 0.99), ls='-')
        max_r = max(max_r, r_all)

    # Паддинг и лимиты
    pad = 200
    lim = max_r + pad
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    # Только нижняя и левая оси
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.set_xlabel('(m)')
    ax.set_ylabel('(m)')

    # Легенда сверху, вне осей
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.2),
        frameon=True,  # рамка
        fancybox=True,  # скруглённые углы
        shadow=False,
        edgecolor='black',
        framealpha=0.8,
        ncol=1,  # один столбик
        borderpad=0.5
    )

    # Добавить пространство сверху под легенду
    fig.subplots_adjust(top=0.82)

    # plt.savefig(fname, bbox_inches='tight')
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    print(f"сид: {seed}")
    # без коллизий
    state_nc = simulate(total_bombs=500, allowed_collisions=False, seed=seed)
    xy_nc = np.array([b.r[:2] for b in state_nc.fallens], float)

    # с коллизиями
    state_col = simulate(total_bombs=500, allowed_collisions=True, seed=seed + 1)
    xy_all = np.array([b.r[:2] for b in state_col.fallens], float)
    mask_col = np.array([b.is_collided() for b in state_col.fallens], bool)
    xy_not = xy_all[~mask_col]
    xy_col = xy_all[mask_col]

    print(f"Всего упало: {len(state_col.fallens)} | со столкновениями: {mask_col.sum()}")

    save_static_topview(xy_nc, xy_not, xy_col, xy_all, "vulkano_topview.png")
