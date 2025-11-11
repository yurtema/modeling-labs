import math
import numpy as np
from numpy.random import default_rng
import simpy as sim
from typing import List, Dict, Optional, Tuple

# ----------------- ФИЗИКА -----------------
g = np.array([0.0, 0.0, -9.81])
Z_GROUND = 0.0
rho = 2600.0
k_p = 0.6


def mass_from_radius(R: float, rho_local: float = rho) -> float:
    return rho_local * (4.0 / 3.0) * np.pi * R ** 3


# ----------------- КЛАССЫ -----------------
class Bomb:
    __slots__ = ("idx", "t_erup", "t0", "r", "v", "m", "R",
                 "_t_ground", "_collided", "_landed", "_t_land")

    def __init__(self, idx: int, t_erup: float, r0: np.ndarray,
                 v0: np.ndarray, mass: float, radius: float):
        self.idx = idx
        self.t_erup = t_erup
        self.t0 = t_erup
        self.r = r0.astype(float)
        self.v = v0.astype(float)
        self.m = mass
        self.R = radius
        self._t_ground = None
        self._collided = False
        self._landed = False
        self._t_land = None

    def calc_r(self, t: float) -> np.ndarray:
        dt = t - self.t0
        return self.r + self.v * dt + 0.5 * g * dt * dt

    def calc_v(self, t: float) -> np.ndarray:
        return self.v + g * (t - self.t0)

    def __repr__(self):
        return f"<Bomb idx={self.idx} r={self.r} v={self.v}>"


class SimulationState:
    def __init__(self, total_bombs: int):
        self.flyings: List[Bomb] = []
        self.fallens: List[Bomb] = []
        self.processes: Dict[Bomb, List[sim.Process]] = {}
        self.all_bombs: List[Bomb] = []
        self.total_bombs = total_bombs


# ----------------- ВСПОМОГАТЕЛЬНЫЕ -----------------
def when_ground(b: Bomb):
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
    now = max(b1.t0, b2.t0)
    Rsum = b1.R + b2.R
    r = b1.calc_r(now) - b2.calc_r(now)
    v = b1.calc_v(now) - b2.calc_v(now)
    A = float(np.dot(v, v))
    B = float(2 * np.dot(r, v))
    C = float(np.dot(r, r) - Rsum * Rsum)

    eps_time = 1e-9
    if C <= 0.0:
        if B >= 0.0:
            return None
        return now + eps_time

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

    tg1 = b1._t_ground if b1._t_ground is not None else when_ground(b1)
    tg2 = b2._t_ground if b2._t_ground is not None else when_ground(b2)
    tg1_val = tg1 if tg1 is not None else float("inf")
    tg2_val = tg2 if tg2 is not None else float("inf")

    if tcol < tg1_val - 1e-9 and tcol < tg2_val - 1e-9:
        return tcol
    return None


def calc_collision(t: float, b1: Bomb, b2: Bomb):
    r1, v1 = b1.calc_r(t), b1.calc_v(t)
    r2, v2 = b2.calc_r(t), b2.calc_v(t)
    n = r1 - r2
    L = np.linalg.norm(n)
    if L < 1e-8:
        return v1, v2
    n /= L
    rel_vn = np.dot(v1 - v2, n)
    if rel_vn >= 0.0:
        return v1, v2
    m1, m2 = b1.m, b2.m
    inv_m_sum = (1.0 / m1 + 1.0 / m2)
    J = -(1.0 + k_p) * rel_vn / inv_m_sum
    v1p = v1 + (J / m1) * n
    v2p = v2 - (J / m2) * n
    return v1p, v2p


def clear_queue(state: SimulationState, b: Bomb):
    if b not in state.processes:
        return
    plist = list(state.processes[b])
    for p in plist:
        for other, other_list in state.processes.items():
            if other is b:
                continue
            if p in other_list:
                try:
                    other_list.remove(p)
                except ValueError:
                    pass
        if not p.triggered:
            try:
                p.interrupt()
            except RuntimeError:
                pass
    state.processes[b].clear()


# ----------------- ПРОЦЕССЫ -----------------
def ground(env, dt, b, state, done_event):
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


def fmt(v):
    return np.array2string(v, precision=2, suppress_small=True, floatmode='fixed')




# ----------------- ОСНОВНАЯ СИМУЛЯЦИЯ -----------------
def simulate_and_record(bombs: List[Bomb], dt_rec=0.02, save_path="flight_data.npz"):
    env = sim.Environment()
    state = SimulationState(total_bombs=len(bombs))
    done_event = env.event()
    state.all_bombs = bombs
    state.flyings = bombs.copy()

    # планируем падения и столкновения
    for b in bombs:
        b._t_ground = when_ground(b)
        if b._t_ground is not None:
            p = env.process(ground(env, b._t_ground - env.now, b, state, done_event))
            state.processes.setdefault(b, []).append(p)

    for i, b1 in enumerate(bombs):
        for b2 in bombs[i + 1:]:
            tcol = when_collision(b1, b2)
            if tcol is not None:
                p = env.process(collision(env, tcol - env.now, b1, b2, state, done_event))
                state.processes.setdefault(b1, []).append(p)
                state.processes.setdefault(b2, []).append(p)

    # запись кадров
    records, times = [], []

    def recorder():
        try:
            while True:
                frame = np.full((len(bombs), 3), np.nan)
                for b in bombs:
                    frame[b.idx] = b.calc_r(env.now) if not b._landed else b.r
                records.append(frame)
                times.append(env.now)
                if done_event.triggered:
                    break
                yield env.timeout(dt_rec)
        except sim.Interrupt:
            return

    env.process(recorder())
    env.run()

    positions = np.asarray(records)
    np.savez_compressed(save_path, positions=positions, times=np.array(times))
    print(f"Saved {len(records)} frames to {save_path}")
    return save_path, state
def fmt(v):
    return np.array2string(v, precision=2, suppress_small=True, floatmode='fixed')






def collision(env, dt, b1, b2, state, done_event):
    try:
        yield env.timeout(max(0.0, dt))
    except sim.Interrupt:
        return
    if (b1 not in state.flyings) or (b2 not in state.flyings):
        return
    t = env.now

    # --- DEBUG INFO BEFORE COLLISION ---
    v1_before, v2_before = b1.calc_v(t), b2.calc_v(t)
    r1_before, r2_before = b1.calc_r(t), b2.calc_r(t)

    print(f"\n=== COLLISION @ t={t:.4f} ===")
    print(f"Bombs {b1.idx} <-> {b2.idx}")
    print(f"Positions before:\n  b1={fmt(r1_before)}\n  b2={fmt(r2_before)}")
    print(f"Velocities before:\n  b1={fmt(v1_before)} (|v|={np.linalg.norm(v1_before):.2f})\n  b2={fmt(v2_before)} (|v|={np.linalg.norm(v2_before):.2f})")

    # --- IMPULSE BEFORE ---
    p_before = b1.m * v1_before + b2.m * v2_before
    print(f"Total momentum before: {fmt(p_before)} |p|={np.linalg.norm(p_before):.2f}")

    # --- COLLISION CALC ---
    v1p, v2p = calc_collision(t, b1, b2)
    r1c, r2c = b1.calc_r(t), b2.calc_r(t)
    b1.r, b1.v, b1.t0 = r1c, v1p, t
    b2.r, b2.v, b2.t0 = r2c, v2p, t
    b1._collided = True
    b2._collided = True

    # --- IMPULSE AFTER ---
    p_after = b1.m * v1p + b2.m * v2p
    print(f"Velocities after:\n  b1={fmt(v1p)} (|v|={np.linalg.norm(v1p):.2f})\n  b2={fmt(v2p)} (|v|={np.linalg.norm(v2p):.2f})")
    print(f"Total momentum after:  {fmt(p_after)} |p|={np.linalg.norm(p_after):.2f}")

    # --- CHECK DIFFERENCES ---
    dp = p_after - p_before
    print(f"Δp = {fmt(dp)}, |Δp|={np.linalg.norm(dp):.6f}")
    print(f"Δv norms: {np.linalg.norm(v1p - v1_before):.2f}, {np.linalg.norm(v2p - v2_before):.2f}")
    print("====================================")

    # --- QUEUE AND NEXT EVENTS ---
    clear_queue(state, b1)
    clear_queue(state, b2)

    b1._t_ground = when_ground(b1)
    b2._t_ground = when_ground(b2)
    if b1._t_ground is not None:
        p1 = env.process(ground(env, b1._t_ground - env.now, b1, state, done_event))
        state.processes.setdefault(b1, []).append(p1)
    if b2._t_ground is not None:
        p2 = env.process(ground(env, b2._t_ground - env.now, b2, state, done_event))
        state.processes.setdefault(b2, []).append(p2)

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


# ----------------- ТЕСТОВЫЙ ЗАПУСК -----------------
if __name__ == "__main__":
    # Создание вручную
    bombs = []
    bombs.append(Bomb(
        idx=0, t_erup=0.0,
        r0=np.array([0.5, 0.0, 2200.0]),
        v0=np.array([0.0, 0.0, 300.0]),
        mass=mass_from_radius(1.0), radius=1.0
    ))
    bombs.append(Bomb(
        idx=1, t_erup=0.0,
        r0=np.array([0.0, 0.0, 3000.0]),
        v0=np.array([0.0, 0.0, -100.0]),
        mass=mass_from_radius(1.0), radius=1.0
    ))

    simulate_and_record(bombs, dt_rec=0.01, save_path="test_flight_data.npz")
