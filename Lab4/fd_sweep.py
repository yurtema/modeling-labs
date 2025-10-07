# fd_sweep.py
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # <— гарантируем безоконный бекенд
import matplotlib.pyplot as plt


# ---- скопированы базовые параметры (держите их в синхроне с intersection_ca.py) ----
L = 50
NS_GREEN = 20
EW_GREEN = 20
STAB_STEPS = 200
STUDY_STEPS = 300

# геометрия перекрестка 2x2 (должно совпадать с игровым файлом)
X_LEFT  = L//2 - 1
X_RIGHT = L//2
Y_TOP   = L//2 - 1
Y_BOT   = L//2

# ----------------- Модель (минимум кода из intersection_ca.py) -----------------
class Car:
    __slots__ = ("pos","stage","cid")
    def __init__(self, pos, stage, cid): self.pos, self.stage, self.cid = pos, stage, cid

class Lane:
    __slots__ = ("L","dir","cars")
    def __init__(self, length, forward):
        self.L = length
        self.dir = +1 if forward else -1
        self.cars = []
    def ahead(self, i): return (i+1) % self.L if self.dir==+1 else (i-1) % self.L

def light(t): return 'NS' if (t % (NS_GREEN+EW_GREEN)) < NS_GREEN else 'EW'

def cross_path(name):
    if name == "WE": return [(X_LEFT, Y_TOP),  (X_RIGHT, Y_TOP)]
    if name == "EW": return [(X_RIGHT, Y_BOT), (X_LEFT,  Y_BOT)]
    if name == "NS": return [(X_LEFT,  Y_TOP), (X_LEFT,  Y_BOT)]
    if name == "SN": return [(X_RIGHT, Y_BOT), (X_RIGHT, Y_TOP)]
    raise ValueError

def entry_index(name, lane):
    a,_ = cross_path(name)
    if name in ("WE","EW"):
        return (a[0]-1) % lane.L if lane.dir==+1 else (a[0]+1) % lane.L
    else:
        return (a[1]-1) % lane.L if lane.dir==+1 else (a[1]+1) % lane.L

def exit_index(name, lane):
    _,b = cross_path(name)
    if name in ("WE","EW"):
        return (b[0]+1) % lane.L if name=="WE" else (b[0]-1) % lane.L
    else:
        return (b[1]+1) % lane.L if name=="NS" else (b[1]-1) % lane.L

def make_world(N_total, seed=123):
    """
    Создаёт 4 ленты и раскладывает N_total машин равномерно по всем направлениям.
    Порядок направлений: WE (W->E), EW (E->W), NS (N->S), SN (S->N).
    """
    random.seed(seed)
    lanes = [Lane(L, True), Lane(L, False), Lane(L, True), Lane(L, False)]
    per_lane = [N_total//4]*4
    for i in range(N_total % 4): per_lane[i] += 1

    cid = 1
    for lane in lanes:
        n = per_lane.pop(0)
        if n>0:
            gap = max(1, lane.L // n)
            idxs = [(k*gap) % lane.L for k in range(n)]
            random.shuffle(idxs)
            for p in idxs[:n]:
                lane.cars.append(Car(pos=p, stage=0, cid=cid)); cid += 1
    return lanes

def one_step(lanes, t):
    lane_WE, lane_EW, lane_NS, lane_SN = lanes
    who = light(t)
    moved = 0
    exits_cnt = {"WE": 0, "EW": 0, "NS": 0, "SN": 0}

    def occ_lane(lane):
        occ = [False]*lane.L
        for c in lane.cars:
            if c.stage==0: occ[c.pos] = True
        return occ
    occ_WE, occ_EW = occ_lane(lane_WE), occ_lane(lane_EW)
    occ_NS, occ_SN = occ_lane(lane_NS), occ_lane(lane_SN)

    cross_occ = {(X_LEFT,Y_TOP):False,(X_RIGHT,Y_TOP):False,(X_LEFT,Y_BOT):False,(X_RIGHT,Y_BOT):False}
    def mark_inside(name, lane):
        a,b = cross_path(name)
        for c in lane.cars:
            if c.stage==1: cross_occ[a] = True
            if c.stage==2: cross_occ[b] = True
    for nm, ln in (("WE",lane_WE),("EW",lane_EW),("NS",lane_NS),("SN",lane_SN)):
        mark_inside(nm, ln)

    # ======== 1) Выезды ========
    def do_exits(lane, occ, name):
        nonlocal moved, exits_cnt
        ex = exit_index(name, lane)
        for c in lane.cars:
            if c.stage == 2 and not occ[ex]:
                c.stage = 0
                c.pos = ex
                occ[ex] = True
                moved += 1
                exits_cnt[name] += 1

    # >>> ВОТ ЭТИ ВЫЗОВЫ ОБЯЗАТЕЛЬНЫ <<<
    do_exits(lane_WE, occ_WE, "WE")
    do_exits(lane_EW, occ_EW, "EW")
    do_exits(lane_NS, occ_NS, "NS")
    do_exits(lane_SN, occ_SN, "SN")
    # ===================================

    # 2) движение внутри 2×2
    def do_inner(lane, name):
        nonlocal moved
        a,b = cross_path(name)
        for c in lane.cars:
            if c.stage==1 and not cross_occ[b]:
                c.stage=2; cross_occ[a]=False; cross_occ[b]=True; moved += 1
                break
    do_inner(lane_WE,"WE"); do_inner(lane_EW,"EW"); do_inner(lane_NS,"NS"); do_inner(lane_SN,"SN")

    # 3) линейные шаги
    def do_line(lane, occ, name):
        nonlocal moved
        ent = entry_index(name, lane)
        intents = []
        for i,c in enumerate(lane.cars):
            if c.stage!=0: intents.append((i,None)); continue
            if c.pos==ent: intents.append((i,None)); continue
            nxt = lane.ahead(c.pos)
            intents.append((i, None if occ[nxt] else nxt))
        for i,nxt in intents:
            if nxt is None: continue
            c = lane.cars[i]; occ[c.pos]=False; c.pos=nxt; occ[c.pos]=True; moved += 1
    do_line(lane_WE,occ_WE,"WE"); do_line(lane_EW,occ_EW,"EW"); do_line(lane_NS,occ_NS,"NS"); do_line(lane_SN,occ_SN,"SN")

    # 4) въезды
    def do_entry(lane, occ, name, allow):
        nonlocal moved
        if not allow: return
        a,_ = cross_path(name); ent = entry_index(name, lane)
        for c in lane.cars:
            if c.stage==0 and c.pos==ent and not cross_occ[a]:
                occ[c.pos]=False; c.stage=1; cross_occ[a]=True; moved += 1; break
    do_entry(lane_WE,occ_WE,"WE", allow=(who=="EW"))
    do_entry(lane_EW,occ_EW,"EW", allow=(who=="EW"))
    do_entry(lane_NS,occ_NS,"NS", allow=(who=="NS"))
    do_entry(lane_SN,occ_SN,"SN", allow=(who=="NS"))

    return moved, exits_cnt

CYCLE = NS_GREEN + EW_GREEN

def run_sim(N_total, seed=123, study_cycles=12):  # кратно циклу!
    lanes = make_world(N_total, seed=seed)
    # прогрев
    for t in range(STAB_STEPS - (STAB_STEPS % CYCLE)):  # округлим до цикла
        one_step(lanes, t)
    # исследование на целое число циклов
    study_steps = study_cycles * CYCLE

    total_moved = 0
    exits_sum = {"WE":0, "EW":0, "NS":0, "SN":0}
    for k in range(study_steps):
        t = k
        moved, ex = one_step(lanes, t)
        total_moved += moved
        for kname in exits_sum:
            exits_sum[kname] += ex[kname]

    N = float(N_total)
    rho = N / (4.0 * L)                       # плотность на полосу
    vbar = (total_moved / study_steps) / N if N > 0 else 0.0  # как раньше, при желании оставим
    # поток на полосу: суммарные выезды / время / число полос
    q = (sum(exits_sum.values()) / study_steps) / 4.0
    return rho, q, vbar


# ----------------- Скан по плотности + график -----------------
def sweep_and_plot(min_rho=0.02, max_rho=0.95, points=25,
                   seeds=5,
                   save_png="fundamental_diagram.png",
                   save_csv="fundamental_diagram.csv"):
    rhos, qs, vs, Ns = [], [], [], []
    max_cars = 4 * L
    targets = [min_rho + k*(max_rho-min_rho)/(points-1) for k in range(points)]
    for i, rho_target in enumerate(targets):
        N = max(0, min(max_cars, int(round(rho_target * 4 * L))))
        if N == 0:   # точка в нуле тоже полезна
            r, q, v = 0.0, 0.0, 0.0
        else:
            q_acc, v_acc = 0.0, 0.0
            for s in range(seeds):
                r, q, v = run_sim(N, seed=1000 + i*97 + s*13, study_cycles=20)
                q_acc += q; v_acc += v
            q, v = q_acc / seeds, v_acc / seeds
        rhos.append(r); qs.append(q); vs.append(v); Ns.append(N)
        print(f"N={N:3d}  rho={r:.3f}  q={q:.4f}  vbar={v:.3f}")

    # сортировка на всякий случай
    zipped = sorted(zip(rhos, qs, vs, Ns), key=lambda t: t[0])
    rhos, qs, vs, Ns = map(list, zip(*zipped))

    # CSV
    with open(save_csv, "w", encoding="utf-8") as f:
        f.write("N_total,rho_per_lane,q_per_lane,vbar\n")
        for N, r, qv, v in zip(Ns, rhos, qs, vs):
            f.write(f"{N},{r:.6f},{qv:.6f},{v:.6f}\n")

    # Графики
    plt.figure()
    plt.plot(rhos, qs, marker="o", markersize=4, linewidth=1.2)
    plt.xlabel("Плотность ρ (на полосу)")
    plt.ylabel("Поток q (на полосу, на тик)")
    plt.title("Фундаментальная диаграмма q(ρ) (счётчик выездов, усреднение по seed)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_png, dpi=160)

    plt.figure()
    plt.plot(rhos, vs, marker="o", markersize=4, linewidth=1.2)
    plt.xlabel("Плотность ρ (на полосу)")
    plt.ylabel("Средняя скорость v̄")
    plt.title("Средняя скорость v̄(ρ)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("avg_speed_vs_density.png", dpi=160)


if __name__ == "__main__":
    sweep_and_plot()
