# s_nfs_full_report.py
# S-NFS (VDR + anticipation) — комплект под отчёт (русифицированный)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# =====================================================================
#                         М О Д Е Л Ь   S-N F S
# =====================================================================

def snfs_step(x, v, L, vmax, p, q_eff, r, rng):
    """Параллельный шаг S-NFS. Мягкий slow-to-start: v_new -= 1, не v_new=0."""
    N = x.size
    ord_ = np.argsort(x, kind="stable"); x = x[ord_]; v = v[ord_]

    d = (np.roll(x, -1) - x - 1) % L
    v_lead = np.roll(v, -1)

    rv1 = rng.random(N); rv2 = rng.random(N); rv3 = rng.random(N)

    # 1) ускорение
    v_new = np.minimum(v + 1, vmax)

    # 2) anticipation (эффективный зазор)
    use_ant = rv1 < r
    g_eff = d
    g_eff[use_ant] = d[use_ant] + np.maximum(v_lead[use_ant] - 1, 0)
    v_new = np.minimum(v_new, g_eff)

    # 3) soft slow-to-start: при старте с вер-тью q_eff теряем 1, а не приклеиваемся к 0
    s2s_mask = (v == 0) & (v_new > 0) & (rv2 < q_eff)
    v_new[s2s_mask] -= 1
    v_new = np.maximum(v_new, 0)

    # 4) случайное торможение
    rb = (rv3 < p) & (v_new > 0)
    v_new[rb] -= 1

    # 5) безопасность
    v_new = np.minimum(v_new, d)

    # 6) перемещение
    x = (x + v_new) % L

    inv = np.empty_like(ord_); inv[ord_] = np.arange(N)
    return x[inv], v_new[inv]


def _init_state(L, N, vmax, init, rng, q_eff):
    """Инициализация: при очень сильном slow-to-start стартуем равномерно."""
    if (init == "random") and (q_eff < 0.9):
        x = np.sort(rng.choice(L, size=N, replace=False)).astype(np.int16)
        v = rng.integers(0, vmax + 1, size=N, dtype=np.int8)
    else:
        step = max(1, L // max(1, N))
        x = (np.arange(N, dtype=np.int16) * step) % L
        v = np.zeros(N, dtype=np.int8)
    return x, v


def run_snfs(L=100, rho=0.1, vmax=3, p=0.1, q=0.0, r=0.0,
             Ts=150, Te=250, init="random", seed=None, early_stop=True,
             q_mode="s2s"):
    """
    Возвращает (flow, vbar). q_mode:
      - "s2s"   : q — сила slow-to-start (вероятность «притормаживания старта»).
      - "paper" : тот же смысл, просто отдельный ярлык для «бумажного» набора.
    """
    rng = np.random.default_rng(seed)
    N = int(round(rho * L))
    if N <= 0:
        return 0.0, 0.0

    # во всех режимах трактуем q одинаково как «сила медленного старта»
    q_eff = float(q)

    x, v = _init_state(L, N, vmax, init, rng, q_eff)

    # прогрев с возможной ранней остановкой
    v_prev, steady = -1.0, 0
    for _ in range(Ts):
        x, v = snfs_step(x, v, L, vmax, p, q_eff, r, rng)
        if early_stop:
            vm = float(v.mean())
            if abs(vm - v_prev) < 1e-6:
                steady += 1
                if steady >= 30:
                    break
            else:
                steady, v_prev = 0, vm

    # измерение
    v_acc = 0.0
    for _ in range(Te):
        x, v = snfs_step(x, v, L, vmax, p, q_eff, r, rng)
        v_acc += float(v.mean())

    v_avg = v_acc / Te
    return rho * v_avg, v_avg


# ============================== УТИЛИТЫ ====================================

def _save_fig(fig, fname):
    fig.tight_layout()
    fig.savefig(fname, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {fname}")


# =====================================================================
#            Ф У Н Д А М Е Н Т А Л Ь Н Ы Е   Д И А Г Р А М М Ы
# =====================================================================

def fd_grid(vmax=3, init="random", *,
            L=100, Ts=150, Te=250, repeats=3, p=0.1,
            q_vals=(0, 0.5, 0.95), r_vals=(0, 0.5, 1.0),
            rhos=None, style="FAST", outfile="FD_grid.png", seed0=10000,
            smooth_win=5, q_mode="s2s"):
    """
    3×3 панели q×r с кривыми q(ρ). style="FAST"/"QUALITY".
    Рисуем среднюю кривую; облака — в QUALITY.
    """
    if rhos is None:
        rhos = (np.linspace(0.02, 0.95, 120) if style == "FAST"
                else np.linspace(0.01, 0.95, 220))

    dot_c, dot_s, dot_a = ("k", 6, 0.65) if style == "QUALITY" else ("#444", 5, 0.6)
    mean_c, mean_lw, grid_a = "k", 1.2, 0.2
    y_max = 1.2 if vmax == 3 else 0.7

    # сглаживание
    smooth_win = max(1, int(smooth_win)); smooth_win += (smooth_win % 2 == 0)
    kernel = np.ones(smooth_win, dtype=float) / smooth_win

    fig, axes = plt.subplots(3, 3, figsize=(9, 8))
    for i, r in enumerate(r_vals):
        for j, q in enumerate(q_vals):
            ax = axes[i, j]
            means = []
            for idx, rho in enumerate(rhos):
                # адаптация окна на больших плотностях
                Ts_loc, Te_loc, es = Ts, Te, True
                if rho > 0.7:
                    Ts_loc = max(Ts, 400)
                    Te_loc = max(Te, 3*L)
                    es = False
                vals = [run_snfs(L, rho, vmax, p, q, r, Ts_loc, Te_loc, init,
                                 seed=seed0 + 97*rep + 7*idx + 3*i + j,
                                 early_stop=es, q_mode=q_mode)
                        for rep in range(repeats)]
                flows = [flow for flow, _ in vals]
                means.append(np.mean(flows))
                if style == "QUALITY":
                    ax.scatter([rho]*repeats, flows, s=dot_s, c=dot_c,
                               alpha=dot_a, linewidths=0)

            means = np.asarray(means)
            if means.size >= smooth_win and smooth_win > 1:
                means = np.convolve(means, kernel, mode="same")

            ax.plot(rhos, means, color=mean_c, lw=mean_lw)
            ax.set_xlim(0.0, 1.0); ax.set_ylim(-0.02, y_max)
            if i == 2: ax.set_xlabel("Плотность ρ")
            if j == 0: ax.set_ylabel("Поток q")
            ax.set_title(f"q={q}, r={r}", fontsize=10)
            ax.grid(True, alpha=grid_a)

    _save_fig(fig, outfile)


def fd_grid_paper(vmax=3, init="random", *,
                  L=100, Ts=200, Te=50, p=0.1,
                  q_vals=(0.0, 0.5, 0.95), r_vals=(0.0, 0.5, 1.0),
                  n_rho=240, reps=6, q_mode="paper",
                  rho_min=0.02, rho_max=0.99, jitter_rho=0.002,
                  marker_size=8, alpha=0.85,
                  outfile="FD_paper_style.png", seed0=12345):
    """
    «Как в работе»: облака точек, случайные ρ, лёгкий джиттер. Подписи — рус.
    """
    rng = np.random.default_rng(seed0)
    rhos = np.sort(rng.uniform(rho_min, rho_max, size=n_rho))

    fig, axes = plt.subplots(3, 3, figsize=(10, 9))
    for i, r in enumerate(r_vals):
        for j, q in enumerate(q_vals):
            ax = axes[i, j]
            xs, ys = [], []
            for idx, rho in enumerate(rhos):
                for rep in range(reps):
                    rho_j = float(np.clip(rho + rng.normal(0, jitter_rho), rho_min, rho_max))
                    # длиннее на плотных режимах
                    Ts_loc, Te_loc, es = Ts, Te, True
                    if rho_j > 0.7:
                        Ts_loc = max(Ts, 400)
                        Te_loc = max(Te, 3*L)
                        es = False
                    flow, _ = run_snfs(L=L, rho=rho_j, vmax=vmax, p=p, q=q, r=r,
                                       Ts=Ts_loc, Te=Te_loc, init=init,
                                       seed=seed0 + 7919*idx + 97*rep + 11*i + j,
                                       early_stop=es, q_mode=q_mode)
                    xs.append(rho_j); ys.append(flow)
            ax.scatter(xs, ys, s=marker_size, c="k", alpha=alpha, linewidths=0)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.2 if vmax==3 else 0.7)
            if i == 2: ax.set_xlabel("Плотность ρ")
            if j == 0: ax.set_ylabel("Поток q")
            ax.set_title(f"q={q}, r={r}", fontsize=10)
            ax.grid(True, alpha=0.15)

    _save_fig(fig, outfile)


# =====================================================================
#         v̄(ρ),  ρ(q),  x–t   и   Т Е П Л О В А Я   К А Р Т А
# =====================================================================

def speed_vs_density(q, r, *, vmax=3, init="random",
                     L=100, Ts=150, Te=250, p=0.1,
                     rhos=None, repeats=5, style="FAST",
                     out_prefix="vbar_vs_rho", seed0=20000, q_mode="s2s"):
    """Построение зависимостей средней скорости и плотности от потока."""
    if rhos is None:
        rhos = (np.linspace(0.02, 0.95, 110) if style == "FAST"
                else np.linspace(0.01, 0.95, 180))

    v_means, q_means = [], []
    for idx, rho in enumerate(rhos):
        # адаптация окна усреднения при больших плотностях
        Ts_loc, Te_loc, es = Ts, Te, True
        if rho > 0.7:
            Ts_loc = max(Ts, 400)
            Te_loc = max(Te, 3 * L)
            es = False

        pairs = [run_snfs(L, rho, vmax, p, q, r, Ts_loc, Te_loc, init,
                          seed=seed0 + 31 * rep + idx * 13,
                          q_mode=q_mode, early_stop=es)
                 for rep in range(repeats)]
        q_means.append(np.mean([p_[0] for p_ in pairs]))
        v_means.append(np.mean([p_[1] for p_ in pairs]))

    # === v̄(ρ) ===
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rhos, v_means, lw=1.8, color="k")
    ax.set_xlabel("Плотность ρ", fontsize=11)
    ax.set_ylabel(r"Средняя скорость $\overline{v}$", fontsize=11)
    ax.set_title(fr"$\overline{{v}}(\rho)$ | q={q}, r={r}, v_{{max}}={vmax}", fontsize=12)
    ax.grid(alpha=0.3)
    _save_fig(fig, f"{out_prefix}_q{q}_r{r}_v{vmax}_{q_mode}.png")

    # === ρ(q) === (инвертированная ось X для классической формы «справа налево») ===
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rhos, q_means, lw=1.8, color='k')  # ← теперь q(ρ), не наоборот
    ax.set_xlabel("Плотность ρ")
    ax.set_ylabel("Поток q")
    ax.set_xlim(0.0, 1.0)
    ax.set_title(f"$q(\\rho)$ | q={q}, r={r}, v_{{max}}={vmax}")
    ax.grid(alpha=0.3)
    _save_fig(fig, f"flow_vs_rho_q{q}_r{r}_v{vmax}_{q_mode}.png")


def xt_and_heatmap(rho=0.15, *, steps=250, L=200, vmax=3,
                   p=0.1, q=0.0, r=1.0, init="random",
                   seed=123, out_prefix="snfs", q_mode="s2s"):
    rng = np.random.default_rng(seed)
    q_eff = float(q)
    N = int(round(rho * L))
    x, v = _init_state(L, N, vmax, init, rng, q_eff)

    occ = np.zeros((steps, L), dtype=np.uint8)
    vel = np.full((steps, L), np.nan, dtype=np.float32)

    for t in range(steps):
        occ[t, x] = 1; vel[t, x] = v
        x, v = snfs_step(x, v, L, vmax, p, q_eff, r, rng)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.imshow(occ, aspect='auto', origin='lower', cmap="gray_r")
    ax.set_xlabel("пространство x"); ax.set_ylabel("время t")
    ax.set_title(f"x–t | ρ={rho}, q={q}, r={r}, v_max={vmax}")
    _save_fig(fig, f"{out_prefix}_xt_r{rho}_q{q}_r{r}_v{vmax}.png")

    fig, ax = plt.subplots(figsize=(8, 3))
    im = ax.imshow(vel, aspect='auto', origin='lower')
    ax.set_xlabel("пространство x"); ax.set_ylabel("время t")
    ax.set_title(f"Тепловая карта скорости | ρ={rho}, q={q}, r={r}, v_max={vmax}")
    fig.colorbar(im, ax=ax, label="скорость")
    _save_fig(fig, f"{out_prefix}_heat_r{rho}_q{q}_r{r}_v{vmax}.png")


# =====================================================================
#                             G I F   А Н И М А Ц И И
# =====================================================================

def gif_loop(rho=0.20, *, L=120, steps=300, vmax=3, p=0.1, q=0.0, r=1.0,
             seed=7, outfile="traffic_loop.gif", q_mode="s2s"):
    rng = np.random.default_rng(seed)
    q_eff = float(q)
    N = int(round(rho * L))
    x, v = _init_state(L, N, vmax, "random", rng, q_eff)

    fig, ax = plt.subplots(figsize=(6, 2))
    ax.set_xlim(0, L); ax.set_ylim(-0.5, 0.5); ax.axis('off')
    scat = ax.scatter(x, np.zeros_like(x), s=16)
    ax.set_title("Кольцо (loop)")

    def update(_):
        nonlocal x, v
        x, v = snfs_step(x, v, L, vmax, p, q_eff, r, rng)
        scat.set_offsets(np.c_[x, np.zeros_like(x)])
        return scat,

    FuncAnimation(fig, update, frames=steps, interval=40, blit=True).save(
        outfile, writer=PillowWriter(fps=25))
    plt.close(fig); print(f"[saved] {outfile}")


def gif_no_loop(alpha=0.45, beta=0.7, *, L=160, steps=350, vmax=3,
                p=0.1, q=0.0, r=1.0, seed=11, outfile="traffic_no_loop.gif",
                q_mode="s2s"):
    rng = np.random.default_rng(seed)
    q_eff = float(q)
    occ = np.zeros(L, dtype=np.uint8); vel = np.zeros(L, dtype=np.int8)

    fig, ax = plt.subplots(figsize=(7, 2))
    ax.set_xlim(0, L); ax.set_ylim(-0.5, 0.5); ax.axis('off')
    pts, = ax.plot([], [], 'o', ms=4)
    ax.set_title(f"Открытая дорога (α={alpha}, β={beta})")

    def step_open(occ, vel):
        x = np.flatnonzero(occ)
        if x.size:
            v = vel[x]
            d = np.empty_like(x)
            d[:-1] = x[1:] - x[:-1] - 1
            d[-1] = (L - 1) - x[-1]
            rv1, rv2, rv3 = rng.random(x.size), rng.random(x.size), rng.random(x.size)
            v_new = np.minimum(v + 1, vmax)
            v_lead = np.empty_like(v); v_lead[:-1] = v[1:]; v_lead[-1] = 0
            g_eff = d + np.where(rv1 < r, np.maximum(v_lead - 1, 0), 0)
            v_new = np.minimum(v_new, g_eff)
            still = (v == 0) & (v_new > 0) & (rv2 < q_eff); v_new[still] -= 1
            v_new = np.maximum(v_new, 0)
            rb = (rv3 < p) & (v_new > 0); v_new[rb] -= 1
            v_new = np.minimum(v_new, d)
            occ[:] = 0; vel[:] = 0
            new_pos = np.minimum(x + v_new, L - 1)
            occ[new_pos] = 1; vel[new_pos] = v_new
            if occ[L-1] and (rng.random() < beta): occ[L-1]=0; vel[L-1]=0
        if (not occ[0]) and (rng.random() < alpha): occ[0]=1; vel[0]=0
        return occ, vel

    def update(_):
        nonlocal occ, vel
        occ, vel = step_open(occ, vel)
        xs = np.flatnonzero(occ); pts.set_data(xs, np.zeros_like(xs))
        return pts,

    FuncAnimation(fig, update, frames=steps, interval=40, blit=True).save(
        outfile, writer=PillowWriter(fps=25))
    plt.close(fig); print(f"[saved] {outfile}")


# =====================================================================
#                              П Р И М Е Р Ы
# =====================================================================

if __name__ == "__main__":
    # 1) FD в paper-стиле (облака)
    fd_grid_paper(vmax=3, init="random",
                  q_mode="paper",
                  Ts=120, Te=40,
                  n_rho=180, reps=4,
                  outfile="FD_vmax3_paper.png")

    fd_grid_paper(vmax=1, init="uniform",
                  q_mode="paper",
                  Ts=120, Te=40,
                  n_rho=180, reps=4,
                  outfile="FD_vmax1_paper.png")

    # 2) Версия с линиями (усреднение) — тоже на русском
    fd_grid(vmax=3, init="random",
            style="QUALITY",
            q_mode="paper",
            outfile="FD_vmax3_quality_paper.png")

    # 3) v̄(ρ) и ρ(q) (S-форма читается справа-налево)
    speed_vs_density(q=0.0, r=1.0, vmax=3, init="random",
                     out_prefix="vbar_vs_rho", q_mode="s2s")
    speed_vs_density(q=0.5, r=1.0, vmax=3, init="random",
                     out_prefix="vbar_vs_rho", q_mode="s2s")

    # 4) x–t и теплокарта (ρ=0.15, r=1)
    xt_and_heatmap(rho=0.15, L=200, vmax=3, q=0.0, r=1.0,
                   out_prefix="snfs_v3_q0_r1", q_mode="s2s")

    # 5) GIF (loop / no-loop)
    gif_loop(rho=0.20, L=120, vmax=3, q=0.0, r=1.0,
             outfile="traffic_loop.gif", q_mode="s2s")
    gif_no_loop(alpha=0.45, beta=0.7, L=160, vmax=3, q=0.0, r=1.0,
                outfile="traffic_no_loop.gif", q_mode="s2s")
