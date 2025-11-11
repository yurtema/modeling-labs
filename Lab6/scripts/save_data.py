import math
import numpy as np
from numpy.random import Generator, default_rng
import random
import simpy as sim
from typing import List, Dict, Optional, Tuple

# ----------------- ФИЗИКА / ПАРАМЕТРЫ (константы) -----------------
g = np.array([0.0, 0.0, -9.81])
H = 2500.0  # Высота вулкана
Dc = 200.0  # Глубина кратера от вершины
Rc = 30.0  # Радиус кратера

Z_GROUND = 0.0

rho = 2600.0
k_p = 0.6


mu_v, sigma_v = 90.0, 12.0
mu_th, sigma_th = np.deg2rad(2.0), np.deg2rad(2.0)
R_min, R_max = 0.5, 1.2
beta_erup = 0.5
n_per_erup = 1
seed = random.randint(0, 1000)


def mass_from_radius(R: float, rho_local: float = rho) -> float:
    return rho_local * (4.0 / 3.0) * np.pi * R ** 3


class Bomb:
    __slots__ = ("idx", "t_erup", "t0", "r", "v", "m", "R", "_t_ground", "_collided", "_landed", "_t_land")

    def __init__(self, idx: int, t_erup: float, r0: np.ndarray, v0: np.ndarray, mass: float, radius: float):
        self.idx = int(idx)
        self.t_erup = float(t_erup)
        self.t0 = float(t_erup)
        self.r = r0.astype(float)
        self.v = v0.astype(float)
        self.m = float(mass)
        self.R = float(radius)
        self._t_ground: Optional[float] = None
        self._collided = False
        self._landed = False
        self._t_land: Optional[float] = None

    def calc_r(self, t: float) -> np.ndarray:
        dt = t - self.t0
        return self.r + self.v * dt + 0.5 * g * dt * dt

    def calc_v(self, t: float) -> np.ndarray:
        return self.v + g * (t - self.t0)

    def is_collided(self) -> bool:
        return bool(self._collided)

    def is_landed(self) -> bool:
        return bool(self._landed)


class SimulationState:
    def __init__(self, total_bombs: int):
        self.flyings: List[Bomb] = []
        self.fallens: List[Bomb] = []
        self.processes: Dict[Bomb, List[sim.Process]] = {}
        self.all_bombs: List[Bomb] = []  # ordered by creation idx
        self.total_bombs = total_bombs


def sample_initial(rs: Generator, Rc: float,
                   mu_v: float, sigma_v: float,
                   mu_th: float, sigma_th: float,
                   R_min: float, R_max: float) -> Tuple[np.ndarray, np.ndarray, float, float]:
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

    R = float(rs.uniform(R_min, R_max))
    m = mass_from_radius(R)
    return r0, v0, m, R


def when_ground(b: Bomb):
    """Время достижения Z_GROUND (абсолютное)."""
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
    Время столкновения (абсолютное), если произойдёт ДО падения.
    Исправления:
      - не требуем наличия t_ground у обеих бомб (если у одной нет — считаем её бесконечно летающей)
      - корректная обработка C <= 0 (перекрытие): считаем столкновение сейчас (now + eps)
      - стабильные пороги для вырожденных случаев
    """
    now = max(b1.t0, b2.t0)
    Rsum = b1.R + b2.R
    r = b1.calc_r(now) - b2.calc_r(now)
    v = b1.calc_v(now) - b2.calc_v(now)
    A = float(np.dot(v, v))
    B = float(2 * np.dot(r, v))
    C = float(np.dot(r, r) - Rsum * Rsum)

    eps_time = 1e-9
    # если уже перекрываются в момент now
    if C <= 0.0:
        # если расходятся (B >= 0) — нет нового столкновения
        if B >= 0.0:
            return None
        # иначе — считаем столкновение "сейчас"
        return now + eps_time

    # если относительная скорость нулевая — нигде не столкнутся (перемещение по невозвратной траектории)
    if A < 1e-14:
        return None

    D = B * B - 4 * A * C
    if D < 0:
        return None
    sD = math.sqrt(D)
    taus = [(-B - sD) / (2 * A), (-B + sD) / (2 * A)]
    taus = [tau for tau in taus if tau > eps_time]
    if not taus:
        return None
    tcol = now + min(taus)

    # время приземления — если его нет, считаем +inf (т.е. бомба не падает раньше)
    tg1 = b1._t_ground if b1._t_ground is not None else when_ground(b1)
    tg2 = b2._t_ground if b2._t_ground is not None else when_ground(b2)
    tg1_val = tg1 if tg1 is not None else float("inf")
    tg2_val = tg2 if tg2 is not None else float("inf")

    # коллизия имеет смысл только если произойдёт до приземления обоих бомб
    if tcol < tg1_val - 1e-9 and tcol < tg2_val - 1e-9:
        return tcol
    return None


def calc_collision(t: float, b1: Bomb, b2: Bomb):
    """
    Импульсное столкновение с коэффициентом восстановления k_p.
    Поправки:
      - стабилизация нормали (если расстояние очень мало, избегаем деления на ноль)
      - защита от уже разлетающихся пар по нормали (rel_vn >= 0)
    """
    r1, v1 = b1.calc_r(t), b1.calc_v(t)
    r2, v2 = b2.calc_r(t), b2.calc_v(t)
    n = r1 - r2
    L = np.linalg.norm(n)
    if L < 1e-8:
        # слишком малый разрыв — не применяем импульс (слишком нестабильно)
        return v1, v2
    n /= L

    rel_vn = np.dot(v1 - v2, n)
    if rel_vn >= 0.0:
        return v1, v2

    m1, m2 = b1.m, b2.m
    inv_m_sum = (1.0 / m1 + 1.0 / m2)
    if inv_m_sum == 0.0:
        return v1, v2

    J = -(1.0 + k_p) * rel_vn / inv_m_sum
    v1p = v1 + (J / m1) * n
    v2p = v2 - (J / m2) * n
    return v1p, v2p


def clear_queue(state: SimulationState, b: Bomb):
    """
    Прерывает и удаляет все процессы, связанные с бомбой `b`.
    Исправление: при удалении процесса очищаем ссылки на него из списков других бомб,
    чтобы не оставалось "висячих" процесс-объектов.
    """
    if b not in state.processes:
        return
    # копируем список, т.к. будем модифицировать state.processes
    plist = list(state.processes[b])
    for p in plist:
        # удаляем p из списков других бомб
        for other, other_list in state.processes.items():
            if other is b:
                continue
            if p in other_list:
                try:
                    other_list.remove(p)
                except ValueError:
                    pass
        # прерываем процесс, если он ещё не выполнен
        if not p.triggered:
            try:
                p.interrupt()
            except RuntimeError:
                pass
    state.processes[b].clear()


def gen_bombs(env: sim.Environment, n: int, rs: Generator, state: SimulationState) -> List[Bomb]:
    bombs = []
    base_idx = len(state.all_bombs)
    for i in range(n):
        r0, v0, m, R = sample_initial(rs, Rc, mu_v, sigma_v, mu_th, sigma_th, R_min, R_max)
        b = Bomb(idx=base_idx + i, t_erup=env.now, r0=r0, v0=v0, mass=m, radius=R)
        bombs.append(b)
        state.all_bombs.append(b)
    return bombs


def ground(env: sim.Environment, dt: float, b: Bomb, state: SimulationState, done_event: sim.Event):
    try:
        yield env.timeout(max(0.0, dt))
    except sim.Interrupt:
        return
    clear_queue(state, b)
    b._landed = True
    b.r = b.calc_r(env.now)
    b.r[2] = Z_GROUND
    b.v = b.calc_v(env.now)
    b.t0 = env.now
    b._t_land = env.now
    if b in state.flyings:
        state.flyings.remove(b)
    state.fallens.append(b)
    if len(state.fallens) >= state.total_bombs and not done_event.triggered:
        done_event.succeed()


def collision(env: sim.Environment, dt: float, b1: Bomb, b2: Bomb, state: SimulationState, done_event: sim.Event):
    try:
        yield env.timeout(max(0.0, dt))
    except sim.Interrupt:
        return
    # если одна из бомб уже не в воздухе — игнорируем
    if (b1 not in state.flyings) or (b2 not in state.flyings):
        return
    t = env.now
    v1p, v2p = calc_collision(t, b1, b2)
    r1c, r2c = b1.calc_r(t), b2.calc_r(t)
    # если calc_collision вернул те же векторы — возможно либо вырожденный случай, либо расходятся
    b1.r, b1.v, b1.t0 = r1c, v1p, t
    b2.r, b2.v, b2.t0 = r2c, v2p, t
    b1._collided = True
    b2._collided = True

    # очистка старых процессов обеих бомб (прерываем все ожидающие события, пересчитаем)
    clear_queue(state, b1)
    clear_queue(state, b2)

    # планируем их новые времена падения (если есть)
    b1._t_ground = when_ground(b1)
    b2._t_ground = when_ground(b2)
    if b1._t_ground is not None:
        p1 = env.process(ground(env, b1._t_ground - env.now, b1, state, done_event))
        state.processes.setdefault(b1, []).append(p1)
    if b2._t_ground is not None:
        p2 = env.process(ground(env, b2._t_ground - env.now, b2, state, done_event))
        state.processes.setdefault(b2, []).append(p2)

    # после столкновения ищем новые возможные столкновения для обоих относительно других бомб
    for a in (b1, b2):
        if a not in state.flyings:
            continue
        for other in list(state.flyings):
            if other is a:
                continue
            tnext = when_collision(a, other)
            if tnext is not None:
                p = env.process(collision(env, tnext - env.now, a, other, state, done_event))
                state.processes.setdefault(a, []).append(p)
                state.processes.setdefault(other, []).append(p)


def eruption_process(env: sim.Environment, rs: Generator, state: SimulationState,
                     allowed_collisions: bool, total_bombs: int, done_event: sim.Event):
    created = 0
    while created < total_bombs:
        dt = rs.exponential(beta_erup)
        yield env.timeout(dt)
        n_to_create = min(n_per_erup, total_bombs - created)
        new_bombs = gen_bombs(env, n_to_create, rs, state)
        created += n_to_create

        for b in new_bombs:
            state.flyings.append(b)
            state.processes.setdefault(b, [])
            b._t_ground = when_ground(b)
            if b._t_ground is not None:
                state.processes[b].append(env.process(ground(env, b._t_ground - env.now, b, state, done_event)))
            if not allowed_collisions:
                continue
            # проверяем столкновения новой бомбы с уже летающими
            for other in list(state.flyings):
                if other is b:
                    continue
                tcol = when_collision(b, other)
                if tcol is not None:
                    p = env.process(collision(env, tcol - env.now, b, other, state, done_event))
                    state.processes[b].append(p)
                    state.processes.setdefault(other, []).append(p)


def recorder(env: sim.Environment, dt_rec: float, state: SimulationState, total_bombs: int, records: List[np.ndarray], times: List[float], done_event: sim.Event):
    try:
        while True:
            frame = np.full((total_bombs, 3), np.nan, dtype=float)
            for b in state.all_bombs:
                if b.is_landed():
                    frame[b.idx, :] = b.r.copy()
                else:
                    if env.now >= b.t0:
                        frame[b.idx, :] = b.calc_r(env.now)
                    else:
                        frame[b.idx, :] = np.nan
            records.append(frame)
            times.append(float(env.now))
            if done_event.triggered:
                break
            yield env.timeout(dt_rec)
    except sim.Interrupt:
        return


def simulate_and_record(total_bombs: int, seed: int = seed, dt_rec: float = 0.02, save_path: str = "flight_data.npz", allowed_collisions: bool = True):
    env = sim.Environment()
    rs = default_rng(seed)
    state = SimulationState(total_bombs=total_bombs)

    done_event = env.event()

    records: List[np.ndarray] = []
    times: List[float] = []

    env.process(eruption_process(env, rs, state, allowed_collisions, total_bombs, done_event))
    env.process(recorder(env, dt_rec, state, total_bombs, records, times, done_event))

    env.run()

    positions = np.asarray(records, dtype=float)
    times = np.asarray(times, dtype=float)

    t_erup = np.full((total_bombs,), np.nan, dtype=float)
    t_land = np.full((total_bombs,), np.nan, dtype=float)
    collided = np.zeros((total_bombs,), dtype=bool)
    radii = np.full((total_bombs,), np.nan, dtype=float)
    masses = np.full((total_bombs,), np.nan, dtype=float)

    for b in state.all_bombs:
        t_erup[b.idx] = float(b.t_erup)
        t_land[b.idx] = b._t_land if (b._t_land is not None) else np.nan
        collided[b.idx] = bool(b._collided)
        radii[b.idx] = b.R
        masses[b.idx] = b.m

    np.savez_compressed(save_path,
                        positions=positions,
                        times=times,
                        t_erup=t_erup,
                        t_land=t_land,
                        collided=collided,
                        radii=radii,
                        masses=masses)
    print(f"Saved simulation to {save_path}")
    print(f"positions.shape = {positions.shape}, n_bombs = {total_bombs}, n_frames = {positions.shape[0]}")
    return save_path, state


if __name__ == "__main__":
    path, state = simulate_and_record(total_bombs=200, seed=0, dt_rec=0.5, save_path="flight_data.npz")
