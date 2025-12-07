
import os
import numpy as np
import matplotlib.pyplot as plt

from latticegas import LatticeGas


def main():
    """
    Запускает расчёт решёточного газа и автоматически
    сохраняет картинки поля скорости каждые save_every шагов.
    """

    # ---------------- ПАРАМЕТРЫ ДЛЯ ВАРИАНТА 3 ----------------
    # Здесь нужно подставить те же значения, которые используются в задании.
    # Если в методичке у тебя u_lb и Re считаются через N=15, просто
    # подставь сюда их численные значения.
    parametrs = {
        "nx": 420,     # число узлов по x
        "ny": 180,     # число узлов по y
        "u_lb": 0.048, # скорость в решёточных единицах (замени на свою при необходимости)
        "Re": 440,     # число Рейнольдса (замени на своё при необходимости)
    }

    # Положение и радиус цилиндра (подстрой под свою постановку)
    obstacle = {
        "xc": 105,   # координата центра по x
        "yc": 90,    # координата центра по y
        "r": 15,     # радиус цилиндра в узлах
    }

    # Общее число шагов и шаг между сохранениями картинок
    n_step = 40_000       # сколько шагов всего считать
    save_every = 2_000    # как часто сохранять картинку (каждые 2000 шагов)

    # Директория для картинок
    outdir = "frames_velocity"
    os.makedirs(outdir, exist_ok=True)

    # ---------------- СОЗДАНИЕ МОДЕЛИ ----------------
    model = LatticeGas(parametrs, obstacle)

    # сразу посчитаем макропараметры на шаге 0
    model.update_macro()

    # ---------------- ЦИКЛ ПО ВРЕМЕНИ ----------------
    for step in range(1, n_step + 1):
        model.step()

        if step % save_every == 0:
            # обновляем макропараметры
            model.update_macro()

            # модуль скорости |u|
            vel_abs = np.sqrt(model.u_x ** 2 + model.u_y ** 2)

            # диагностика устойчивости
            finite = np.isfinite(vel_abs)
            if not finite.any():
                print(f"step {step}: ВСЕ значения |u| = NaN/inf, кадр пропущен")
                continue

            data = vel_abs[finite]
            vmin_data = data.min()
            vmax_data = data.max()
            print(
                f"step {step}: min|u|={vmin_data:.5f}, "
                f"max|u|={vmax_data:.5f}, "
                f"NaN count={np.isnan(vel_abs).sum()}"
            )

            # немного расширим диапазон, чтобы был контраст
            if vmax_data > vmin_data:
                dv = 0.05 * (vmax_data - vmin_data)
            else:
                dv = 0.01 * max(vmax_data, 1e-6)
            vmin = max(0.0, vmin_data - dv)
            vmax = vmax_data + dv

            # подготавливаем поле для отрисовки
            vel_plot = vel_abs.T.copy()
            obst = model.obstacle.T
            vel_plot[obst] = np.nan  # внутри цилиндра не рисуем

            # рисуем
            fig, ax = plt.subplots(figsize=(8, 4), dpi=300, tight_layout=True)
            im = ax.imshow(vel_plot, origin="lower", vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=ax, label="|u|")

            ax.set_title(f"Iteration {step}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")

            fname = os.path.join(outdir, f"velocity_{step:06d}.png")
            fig.savefig(fname)
            plt.close(fig)

            print(f"Сохранён кадр {fname}")

    print("Расчёт завершён.")


if __name__ == "__main__":
    main()
