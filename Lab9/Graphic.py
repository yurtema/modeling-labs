import pandas as pd
import matplotlib.pyplot as plt

deposits = [100, 1000, 100000, 1_000_000]

# ---------- 1. Динамика капитала одной игры для каждого депозита ----------

for dep in deposits:
    df = pd.read_csv(f"deposit_{dep}_single.csv")
    plt.figure(figsize=(6, 4))
    plt.step(df["step"], df["capital"], where="post")
    plt.xlabel("Шаг")
    plt.ylabel("Капитал")
    plt.title(f"Динамика капитала (депозит {dep})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"plot_single_{dep}.png", dpi=150)
    plt.close()

# ---------- 2. 20 игр: число шагов до выигрыша/проигрыша -------------------

for dep in deposits:
    df = pd.read_csv(f"deposit_{dep}_20games.csv")
    colors = ["tab:green" if w == 1 else "tab:red" for w in df["win"]]

    plt.figure(figsize=(7, 4))
    plt.bar(df["game"], df["steps"], color=colors)
    plt.xlabel("Игра")
    plt.ylabel("Шаги до результата")
    plt.title(f"20 игр (депозит {dep})")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"plot_20games_{dep}.png", dpi=150)
    plt.close()

# ---------- 3. Статистика по 1000 играм -----------------------------------

stats = pd.read_csv("deposit_stats.csv")

plt.figure(figsize=(6, 4))
plt.bar(stats["deposit"].astype(str),
        stats["wins"] / stats["totalGames"] * 100,
        label="Победы")
plt.bar(stats["deposit"].astype(str),
        stats["losses"] / stats["totalGames"] * 100,
        bottom=stats["wins"] / stats["totalGames"] * 100,
        alpha=0.5,
        label="Поражения")
plt.xlabel("Депозит")
plt.ylabel("Процент игр")
plt.title("Результаты 1000 игр")
plt.legend()
plt.tight_layout()
plt.savefig("plot_stats_win_loss.png", dpi=150)
plt.close()

plt.figure(figsize=(6, 4))
plt.plot(stats["deposit"].astype(str), stats["meanProfit"], marker="o")
plt.xlabel("Депозит")
plt.ylabel("Средний выигрыш от депозита")
plt.title("Средний результат за 1000 игр")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_stats_mean_profit.png", dpi=150)
plt.close()

# ---------- 4. Кинетический метод Монте-Карло (обновлённый) --------------

kmc = pd.read_csv("kmc_traj.csv")

plt.figure(figsize=(7, 4))
plt.plot(kmc["t"], kmc["A"], "r-", label="A(t)")
plt.plot(kmc["t"], kmc["B"], "b-", label="B(t)")
plt.xlabel("t")
plt.ylabel("Количество частиц")
plt.title("Кинетический метод Монте-Карло (Δt = 0.02)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("plot_kmc_AB.png", dpi=150)
plt.close()

