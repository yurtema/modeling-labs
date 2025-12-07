
import numpy as np

class LatticeGas:
    """
    Реализация решёточного газа D2Q9 для лабораторной работы.
    Используется как в самостоятельной программе, так и (при желании) в ноутбуках.
    """

    # ---------- статические вспомогательные методы ----------

    @staticmethod
    def add_cylinder(xc: int, yc: int, radius: int, shape):
        """
        Построить дискретный цилиндр (круг) на решётке.

        Parameters
        ----------
        xc, yc : int
            Координаты центра окружности (в индексах ячеек).
        radius : int
            Радиус окружности в ячейках.
        shape : tuple[int, int]
            Размер решётки (nx, ny).

        Returns
        -------
        dict with keys 'x', 'y' : списки координат занятых ячеек.

        Raises
        ------
        ValueError
            Если цилиндр слишком близко к границам (не помещается
            с запасом в одну ячейку).
        """
        nx, ny = shape

        # Окружность должна целиком лежать внутри области
        # и не касаться внешних границ.
        if (xc - radius) <= 0 or (yc - radius) <= 0 \
           or (xc + radius) >= nx - 1 or (yc + radius) >= ny - 1:
            raise ValueError("Cylinder is too close to the boundary")

        xs = []
        ys = []
        for x in range(nx):
            for y in range(ny):
                if (x - xc) ** 2 + (y - yc) ** 2 <= radius ** 2:
                    xs.append(x)
                    ys.append(y)
        return {"x": xs, "y": ys}

    @staticmethod
    def calc_outflow(f_in: np.ndarray) -> None:
        """
        Правое граничное условие (outflow) на последнем столбце.
        Меняет f_in *на месте*.

        На правой границе входящие в область скорости (v_x < 0)
        берутся равными противоположным им исходящим (v_x > 0).
        Для выбранной нумерации направлений пары:
          0 <-> 8, 1 <-> 7, 2 <-> 6.
        """
        x_last = f_in.shape[1] - 1
        pairs = [(0, 8), (1, 7), (2, 6)]  # (src (v_x>0), dst (v_x<0))
        for src, dst in pairs:
            f_in[dst, x_last, :] = f_in[src, x_last, :]

    @staticmethod
    def calc_u(density: np.ndarray,
               f_in: np.ndarray,
               v: np.ndarray) -> np.ndarray:
        """
        Расчёт макроскопической скорости по формулам (6)-(7) методички.

        Parameters
        ----------
        density : (nx, ny, 1)
        f_in    : (N, nx, ny, 1)
        v       : (N, 2)

        Returns
        -------
        u : (nx, ny, 2)
        """
        rho = density[..., 0]  # (nx, ny)
        nx, ny = rho.shape
        u = np.zeros((nx, ny, 2), dtype=float)

        nonzero = rho != 0
        if not np.any(nonzero):
            return u

        for k in range(2):
            num = np.sum(f_in[:, :, :, 0] * v[:, k][:, None, None], axis=0)
            tmp = np.zeros_like(rho)
            tmp[nonzero] = num[nonzero] / rho[nonzero]
            u[:, :, k] = tmp
        return u

    # ---------- конструктор и основные поля ----------

    def __init__(self, parametrs: dict, obstacle: dict):
        """
        parametrs: {'nx', 'ny', 'u_lb', 'Re'}
        obstacle: {'xc', 'yc', 'r'}
        """
        self.nx = int(parametrs["nx"])
        self.ny = int(parametrs["ny"])
        self.u_lb = float(parametrs["u_lb"])
        self.Re = float(parametrs["Re"])

        # Векторы скоростей (3) и веса a_i (D2Q9)
        self._v = np.array(
            [
                [1.0,  1.0],   # 0: NE
                [1.0,  0.0],   # 1: E
                [1.0, -1.0],   # 2: SE
                [0.0,  1.0],   # 3: N
                [0.0,  0.0],   # 4: rest
                [0.0, -1.0],   # 5: S
                [-1.0, 1.0],   # 6: NW
                [-1.0, 0.0],   # 7: W
                [-1.0,-1.0],   # 8: SW
            ],
            dtype=float,
        )
        self._vx = self._v[:, 0]
        self._vy = self._v[:, 1]
        self._index = np.arange(9, dtype=int)

        self._a = np.array(
            [
                1.0 / 36.0,
                1.0 / 9.0,
                1.0 / 36.0,
                1.0 / 9.0,
                4.0 / 9.0,
                1.0 / 9.0,
                1.0 / 36.0,
                1.0 / 9.0,
                1.0 / 36.0,
            ],
            dtype=float,
        )

        # Группы направлений по знаку v_x (можно использовать при inflow)
        self._ind_right = np.where(self._vx > 0)[0]   # 0,1,2
        self._ind_left = np.where(self._vx < 0)[0]    # 6,7,8
        self._ind_middle = np.where(self._vx == 0)[0] # 3,4,5

        # Вязкость и параметр столкновений ω по (4)-(5) с мягкой стабилизацией
        r = float(obstacle["r"])
        self.radius = r
        self.nu = self.u_lb * r / self.Re              # (5)

        # ν = (1/3)(1/ω - 1/2)  →  ω = 1 / (3ν + 1/2)
        # вводим минимальное τ = 3ν + 1/2, чтобы ω не подходил слишком близко к 2
        tau = 3.0 * self.nu + 0.5
        tau_min = 0.6   # можно немного менять (0.6–0.7) при необходимости
        if tau < tau_min:
            tau = tau_min
        self.omega = 1.0 / tau

        # Макропараметры
        self.density = np.ones((self.nx, self.ny), dtype=float)
        self.u = np.zeros((self.nx, self.ny, 2), dtype=float)
        self.u[:, :, 0] = self.u_lb    # начальная скорость вдоль x
        self.u_x = self.u[:, :, 0]
        self.u_y = self.u[:, :, 1]
        self.p = np.zeros((self.nx, self.ny), dtype=float)

        # Функции распределения
        self.f_in = np.zeros((9, self.nx, self.ny), dtype=float)
        self.f_out = np.zeros_like(self.f_in)
        self.f_equil = np.zeros_like(self.f_in)

        # Препятствие (цилиндр)
        xc = int(obstacle["xc"])
        yc = int(obstacle["yc"])
        cyl = self.add_cylinder(xc, yc, int(r), (self.nx, self.ny))
        self.obstacle = np.zeros((self.nx, self.ny), dtype=bool)
        self.obstacle[cyl["x"], cyl["y"]] = True

        # Скорость потока на входе
        self.v_init = self.u_lb

        # Инициализация равновесного состояния и начальных f_in
        self.update_equilibrium()
        self.f_in[:, :, :] = self.f_equil

        # Пулы для сохранения эволюции (если нужно)
        self.field_den = []
        self.field_u = []
        self.field_p = []
        self.field_ux = []
        self.field_uy = []

    # ---------- расчёты равновесия и макропараметров ----------

    def update_equilibrium(self) -> None:
        """
        Пересчитать массив f_equil (равновесные распределения)
        по текущим density и u.
        """
        ux = self.u_x
        uy = self.u_y
        norma = ux ** 2 + uy ** 2

        for i in range(9):
            vx, vy = self._v[i]
            dot = ux * vx + uy * vy
            self.f_equil[i, :, :] = (
                self.density
                * self._a[i]
                * (1.0 + 3.0 * dot + 4.5 * dot ** 2 - 1.5 * norma)
            )

    def calc_f_out(self) -> None:
        """
        Столкновение: вычислить пост-столкновительные распределения f_out
        по формуле (1).
        """
        self.f_out = self.f_in - self.omega * (self.f_in - self.f_equil)

    def update_macro(self) -> None:
        """
        Обновить плотность, скорость и давление по (6)-(8).
        """
        # (6) плотность
        self.density = np.sum(self.f_in, axis=0)

        # (7) скорость
        ux = np.zeros_like(self.density)
        uy = np.zeros_like(self.density)

        for i in range(9):
            fi = self.f_in[i, :, :]
            vx, vy = self._v[i]
            ux += fi * vx
            uy += fi * vy

        nonzero = self.density != 0
        self.u[:, :, 0] = 0.0
        self.u[:, :, 1] = 0.0
        self.u[nonzero, 0] = ux[nonzero] / self.density[nonzero]
        self.u[nonzero, 1] = uy[nonzero] / self.density[nonzero]

        self.u_x = self.u[:, :, 0]
        self.u_y = self.u[:, :, 1]

        # (8) давление
        self.p = self.density / 3.0

    # ---------- граничные условия и шаг по времени ----------

    def calc_inflow(self) -> None:
        """
        Левое граничное условие (inflow) с заданной скоростью u_x = v_init.
        Реализована схема Zou–He для D2Q9 (перенумерованная под наш порядок).
        """
        x0 = 0
        u_x = self.v_init

        # Скорости на левой границе
        self.u_y[x0, :] = 0.0
        self.u_x[x0, :] = u_x

        # Известные распределения на левой границе:
        f4 = self.f_in[4, x0, :]  # rest
        f3 = self.f_in[3, x0, :]  # N
        f5 = self.f_in[5, x0, :]  # S
        f7 = self.f_in[7, x0, :]  # W
        f6 = self.f_in[6, x0, :]  # NW
        f8 = self.f_in[8, x0, :]  # SW

        # Плотность на левой границе (аналог формулы из Zou–He).
        rho = (f4 + f3 + f5 + 2.0 * (f7 + f6 + f8)) / (1.0 - u_x)
        self.density[x0, :] = rho

        # Неизвестные распределения (v_x > 0): 1 (E), 0 (NE), 2 (SE).
        self.f_in[1, x0, :] = f7 + (2.0 / 3.0) * rho * u_x
        self.f_in[0, x0, :] = (
            f8
            + (1.0 / 6.0) * rho * u_x
            + 0.5 * (f3 - f5)
        )
        self.f_in[2, x0, :] = (
            f6
            + (1.0 / 6.0) * rho * u_x
            + 0.5 * (f5 - f3)
        )

    def bounce_back(self) -> None:
        """
        Граничное условие отражения на препятствии (bounce-back).
        Для узлов, принадлежащих цилиндру, распределения отражаются
        в противоположных направлениях.
        """
        mask = self.obstacle
        for i in range(9):
            opp = 8 - i  # v_i = -v_{8-i} при выбранной нумерации
            self.f_out[i, mask] = self.f_in[opp, mask]

    def collision(self) -> None:
        """
        Этап распространения (streaming): перенос f_out вдоль направлений v_i
        с учётом периодических граничных условий по y.
        """
        for i in range(9):
            vx = int(self._vx[i])
            vy = int(self._vy[i])
            tmp = np.roll(self.f_out[i, :, :], shift=vy, axis=1)
            tmp = np.roll(tmp, shift=vx, axis=0)
            self.f_in[i, :, :] = tmp

    def step(self) -> None:
        """
        Один шаг по времени:
          1) outflow на правой границе;
          2) пересчёт макропараметров ρ, u, p;
          3) inflow на левой границе;
          4) обновление f_equil;
          5) столкновение (f_out);
          6) bounce-back на препятствии;
          7) перенос (streaming) -> обновление f_in.
        """
        self.calc_outflow(self.f_in)   # 7.1
        self.update_macro()            # 7.2
        self.calc_inflow()             # 7.3
        self.update_equilibrium()      # f_eq
        self.calc_f_out()              # 7.4
        self.bounce_back()             # 7.5
        self.collision()               # 7.6

    # ---------- метод solve (опционально, для ноутбуков) ----------

    def solve(self, n_step: int, step_frame: int) -> None:
        """
        Запускает n_step шагов.
        Каждые step_frame шагов сохраняет снимки полей
        в списки field_den, field_u, field_p, field_ux, field_uy.
        """
        self.field_den = []
        self.field_u = []
        self.field_p = []
        self.field_ux = []
        self.field_uy = []

        self.update_macro()
        vel_abs = np.sqrt(self.u_x ** 2 + self.u_y ** 2)
        self.field_den.append(self.density.copy())
        self.field_p.append(self.p.copy())
        self.field_u.append(vel_abs.copy())
        self.field_ux.append(self.u_x.copy())
        self.field_uy.append(self.u_y.copy())

        for step in range(1, n_step + 1):
            self.step()
            if step % step_frame == 0:
                self.update_macro()
                vel_abs = np.sqrt(self.u_x ** 2 + self.u_y ** 2)
                self.field_den.append(self.density.copy())
                self.field_p.append(self.p.copy())
                self.field_u.append(vel_abs.copy())
                self.field_ux.append(self.u_x.copy())
                self.field_uy.append(self.u_y.copy())

        self.field_den = np.array(self.field_den)
        self.field_p = np.array(self.field_p)
        self.field_u = np.array(self.field_u)
        self.field_ux = np.array(self.field_ux)
        self.field_uy = np.array(self.field_uy)

    # ---------- вспомогательный метод для тестов ----------

    def calc_f_eq_i(self, i: int,
                    u: np.ndarray,
                    density: np.ndarray) -> np.ndarray:
        ux = u[:, :, 0]
        uy = u[:, :, 1]
        norma = ux ** 2 + uy ** 2
        vx, vy = self._v[i]
        a = self._a[i]
        dot = ux * vx + uy * vy
        f_eq = density * a * (1.0 + 3.0 * dot + 4.5 * dot ** 2 - 1.5 * norma)
        return f_eq
