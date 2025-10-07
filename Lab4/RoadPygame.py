# intersection_ca.py — соосный перекресток 2×2, строгие фазы, ID и маркер направления
import sys, random
import pygame as pg
from dataclasses import dataclass

# ===================== ПАРАМЕТРЫ (описание) =====================
L = 50            # длина каждой ленты (клеток) от края до края; центр — перекресток 2×2
NS_GREEN = 20     # длительность зелёной фазы для потоков север↔юг (в тиках)
EW_GREEN = 20     # длительность зелёной фазы для потоков запад↔восток (в тиках)
TICK_MS = 60     # длительность одного тика визуализации (мс)

CARS_NS = 24      # суммарно машин на вертикальной оси (поровну в N->S и S->N)
CARS_EW = 24      # суммарно машин на горизонтальной оси (поровну в W->E и E->W)

CELL = 14         # размер клетки (px)
PAD  = 8         # отступы от краёв окна
GRID_ON = False    # показывать сетку

# Цвета (можно менять по вкусу)
BG       = (25,25,28)
ASPHALT  = (40,44,52)
INT_MARK = (90,95,105)     # заливка 2×2
STOPLINE = (130,135,145)   # стоп-линии
CAR_NS   = (255,199,0)
CAR_EW   = (80,200,255)
GRID     = (60,65,72)
TEXT     = (230,232,236)
RED      = (200,60,60)
GREEN    = (70,200,120)
LANE_HL = (235, 235, 240)   # светлый контур полос "выделение"
LANE_HL_W = 2               # толщина контура в пикселях
CENTERLINE = (235, 235, 235)   # цвет пунктирной осевой
DASH_LEN   = max(2, CELL // 2) # длина штриха
GAP_LEN    = CELL               # длина промежутка
THICK      = max(2, CELL // 6)  # толщина штриха



# Геометрия 2×2 в центре
X_LEFT  = L//2 - 1
X_RIGHT = L//2
Y_TOP   = L//2 - 1
Y_BOT   = L//2

# ===================== ДАННЫЕ =====================
@dataclass
class Car:
    pos: int      # индекс клетки на своей ленте (вне 2×2)
    stage: int    # 0=вне; 1=первая клетка пути в 2×2; 2=вторая клетка пути
    cid: int      # уникальный номер

class Lane:
    def __init__(self, length: int, forward: bool, seed: int):
        self.L = length
        self.dir = +1 if forward else -1
        self.cars: list[Car] = []
        self._seed = seed

    def ahead(self, i: int) -> int:
        return (i + 1) % self.L if self.dir == +1 else (i - 1) % self.L

    def spawn_evenly(self, n: int, cid_start: int) -> int:
        random.seed(self._seed)
        if n <= 0: return cid_start
        gap = max(1, self.L // n)
        idxs = [(k*gap) % self.L for k in range(n)]
        random.shuffle(idxs)
        for p in idxs[:n]:
            self.cars.append(Car(pos=p, stage=0, cid=cid_start))
            cid_start += 1
        return cid_start

# ===================== ВСПОМОГАТЕЛЬНОЕ =====================
def light(t: int) -> str:
    """'NS' или 'EW' — чья фаза зелёная."""
    return 'NS' if (t % (NS_GREEN+EW_GREEN)) < NS_GREEN else 'EW'

def cross_path(name: str):
    """Две клетки (x,y) внутри 2×2, через которые идёт поток данного направления."""
    if name == "WE": return [(X_LEFT, Y_TOP),  (X_RIGHT, Y_TOP)]   # слева→вправо, верхний ряд
    if name == "EW": return [(X_RIGHT, Y_BOT), (X_LEFT,  Y_BOT)]   # справа→влево, нижний ряд
    if name == "NS": return [(X_LEFT,  Y_TOP), (X_LEFT,  Y_BOT)]   # сверху→вниз, левая колонка
    if name == "SN": return [(X_RIGHT, Y_BOT), (X_RIGHT, Y_TOP)]   # снизу→вверх, правая колонка
    raise ValueError

def entry_index(name: str, lane: Lane) -> int:
    """Стоп-линия: из этой клетки следующий шаг попадает в первую клетку 2×2."""
    a,_ = cross_path(name)
    if name in ("WE","EW"):  # горизонтали: pos == x
        return (a[0] - 1) % lane.L if lane.dir == +1 else (a[0] + 1) % lane.L
    else:                    # вертикали: pos == y
        return (a[1] - 1) % lane.L if lane.dir == +1 else (a[1] + 1) % lane.L

def exit_index(name: str, lane: Lane) -> int:
    """Первая линейная клетка после второй 2×2."""
    _,b = cross_path(name)
    if name in ("WE","EW"):
        return (b[0] + 1) % lane.L if name == "WE" else (b[0] - 1) % lane.L
    else:
        return (b[1] + 1) % lane.L if name == "NS" else (b[1] - 1) % lane.L

def make_world():
    cid = 1
    lane_WE = Lane(L, True,  201)   # W->E (верхний ряд)
    lane_EW = Lane(L, False, 202)   # E->W (нижний ряд)
    lane_NS = Lane(L, True,  101)   # N->S (левая колонка)
    lane_SN = Lane(L, False, 102)   # S->N (правая колонка)
    cid = lane_NS.spawn_evenly(CARS_NS//2, cid)
    cid = lane_SN.spawn_evenly(CARS_NS - CARS_NS//2, cid)
    cid = lane_WE.spawn_evenly(CARS_EW//2, cid)
    cid = lane_EW.spawn_evenly(CARS_EW - CARS_EW//2, cid)
    return lane_WE, lane_EW, lane_NS, lane_SN

# ===================== ОДИН ШАГ КА (строгий порядок) =====================
def one_step(lanes, t: int):
    lane_WE, lane_EW, lane_NS, lane_SN = lanes
    who = light(t)

    # Линейная занятость (только stage=0)
    def occ_lane(lane: Lane):
        occ = [False]*lane.L
        for c in lane.cars:
            if c.stage == 0: occ[c.pos] = True
        return occ
    occ_WE, occ_EW = occ_lane(lane_WE), occ_lane(lane_EW)
    occ_NS, occ_SN = occ_lane(lane_NS), occ_lane(lane_SN)

    # 2×2 занятость
    cross_occ = {(X_LEFT,Y_TOP):False,(X_RIGHT,Y_TOP):False,
                 (X_LEFT,Y_BOT):False,(X_RIGHT,Y_BOT):False}
    def mark_inside(name, lane):
        a,b = cross_path(name)
        for c in lane.cars:
            if c.stage == 1: cross_occ[a] = True
            if c.stage == 2: cross_occ[b] = True
    for nm, ln in (("WE",lane_WE),("EW",lane_EW),("NS",lane_NS),("SN",lane_SN)):
        mark_inside(nm, ln)

    moved = 0

    # 1) ВЫХОДЫ: stage=2 -> линия (приоритет, вне зависимости от фазы)
    def do_exits(lane: Lane, occ, name):
        nonlocal moved
        ex = exit_index(name, lane)
        # можно вывести несколько (если точка выезда свободна)
        for c in lane.cars:
            if c.stage == 2 and not occ[ex]:
                c.stage = 0
                c.pos = ex
                occ[ex] = True
                moved += 1
    do_exits(lane_WE, occ_WE, "WE")
    do_exits(lane_EW, occ_EW, "EW")
    do_exits(lane_NS, occ_NS, "NS")
    do_exits(lane_SN, occ_SN, "SN")

    # 2) ВНУТРИ 2×2: stage=1 -> stage=2 (если вторая клетка свободна)
    def do_inner(lane: Lane, name):
        nonlocal moved
        a,b = cross_path(name)
        # один шаг внутри с ленты за тик (можно сделать и «все», но этого достаточно)
        for c in lane.cars:
            if c.stage == 1 and not cross_occ[b]:
                c.stage = 2
                cross_occ[a] = False
                cross_occ[b] = True
                moved += 1
                break
    do_inner(lane_WE, "WE")
    do_inner(lane_EW, "EW")
    do_inner(lane_NS, "NS")
    do_inner(lane_SN, "SN")

    # 3) ЛИНЕЙНЫЕ ХОДЫ ВНЕ 2×2 (с блоком на стоп-линии!)
    def do_line(lane: Lane, occ, name):
        nonlocal moved
        ent = entry_index(name, lane)
        intents = []
        for i,c in enumerate(lane.cars):
            if c.stage != 0:
                intents.append((i,None)); continue
            if c.pos == ent:
                # !!! На стоп-линии запрещаем «обычный» шаг — вход только в шаге 4 по зелёному
                intents.append((i,None))
                continue
            nxt = lane.ahead(c.pos)
            intents.append((i, None if occ[nxt] else nxt))
        # применяем
        for i,nxt in intents:
            if nxt is None: continue
            c = lane.cars[i]
            occ[c.pos] = False
            c.pos = nxt
            occ[c.pos] = True
            moved += 1
    do_line(lane_WE, occ_WE, "WE")
    do_line(lane_EW, occ_EW, "EW")
    do_line(lane_NS, occ_NS, "NS")
    do_line(lane_SN, occ_SN, "SN")

    # 4) ВЪЕЗДЫ В 2×2 (только на свою зелёную фазу, если свободна первая клетка)
    def do_entry(lane: Lane, occ, name, allow: bool):
        nonlocal moved
        if not allow: return
        a,_ = cross_path(name)
        ent = entry_index(name, lane)
        # на стоп-линии может стоять максимум 1 машина
        for c in lane.cars:
            if c.stage == 0 and c.pos == ent and not cross_occ[a]:
                occ[c.pos] = False
                c.stage = 1
                cross_occ[a] = True
                moved += 1
                break
    do_entry(lane_WE, occ_WE, "WE", allow=(who=="EW"))
    do_entry(lane_EW, occ_EW, "EW", allow=(who=="EW"))
    do_entry(lane_NS, occ_NS, "NS", allow=(who=="NS"))
    do_entry(lane_SN, occ_SN, "SN", allow=(who=="NS"))

    return moved

# ===================== ОТРИСОВКА =====================
def scr_size(): return PAD*2 + L*CELL, PAD*2 + L*CELL
def cell_rect(x,y): return (PAD + x*CELL, PAD + y*CELL, CELL, CELL)

def draw_lane_highlight(screen):
    # 2 горизонтальные полосы (ряд Y_TOP и ряд Y_BOT)
    pg.draw.rect(screen, LANE_HL,
                 (PAD, PAD + Y_TOP*CELL, L*CELL, CELL), width=LANE_HL_W)
    pg.draw.rect(screen, LANE_HL,
                 (PAD, PAD + Y_BOT*CELL, L*CELL, CELL), width=LANE_HL_W)
    # 2 вертикальные полосы (колонка X_LEFT и X_RIGHT)
    pg.draw.rect(screen, LANE_HL,
                 (PAD + X_LEFT*CELL,  PAD, CELL, L*CELL), width=LANE_HL_W)
    pg.draw.rect(screen, LANE_HL,
                 (PAD + X_RIGHT*CELL, PAD, CELL, L*CELL), width=LANE_HL_W)

def draw_dashed_centerlines(screen):
    # центр между полосами
    cx = PAD + ((X_LEFT + X_RIGHT + 1) / 2) * CELL
    cy = PAD + ((Y_TOP  + Y_BOT   + 1) / 2) * CELL

    # геометрия штриха
    dash  = max(4, int(CELL * 0.6))     # длина штриха в клетке
    thick = max(2, CELL // 6)           # толщина линии
    off   = (CELL - dash) // 2          # отступ внутри клетки

    # 1) подчистим подложку вдоль осевых, чтобы пунктир не «слипался» с рамками/стоп-линиями
    # горизонтальная
    pg.draw.rect(
        screen, ASPHALT,
        (PAD, int(cy - thick/2), L*CELL, thick)
    )
    # вертикальная
    pg.draw.rect(
        screen, ASPHALT,
        (int(cx - thick/2), PAD, thick, L*CELL)
    )

    # 2) штрихи по клеткам (каждая вторая клетка — гарантированный зазор)
    # горизонталь
    y = int(cy - thick/2)
    for col in range(0, L, 2):
        x0 = PAD + col * CELL + off
        pg.draw.rect(screen, CENTERLINE, (x0, y, dash, thick))

    # вертикаль
    x = int(cx - thick/2)
    for row in range(0, L, 2):
        y0 = PAD + row * CELL + off
        pg.draw.rect(screen, CENTERLINE, (x, y0, thick, dash))


def lane_cell_rect(name: str, idx: int):
    """Рисуем ровно по индексам сетки: горизонтали на Y_TOP/Y_BOT, вертикали на X_LEFT/X_RIGHT."""
    if name == "WE": return cell_rect(idx, Y_TOP)
    if name == "EW": return cell_rect(idx, Y_BOT)
    if name == "NS": return cell_rect(X_LEFT,  idx)
    if name == "SN": return cell_rect(X_RIGHT, idx)
    raise ValueError

def draw_grid(screen):
    if not GRID_ON: return
    for i in range(L+1):
        x = PAD + i*CELL; y = PAD + i*CELL
        pg.draw.line(screen, GRID, (PAD,y), (PAD+L*CELL,y), 1)
        pg.draw.line(screen, GRID, (x,PAD), (x,PAD+L*CELL), 1)

def draw_stop_lines(screen):
    # Стоп-линии прямо перед входом в первую клетку 2×2
    for (x,y) in [(X_LEFT-1, Y_TOP), (X_RIGHT+1, Y_BOT),
                  (X_LEFT,  Y_TOP-1), (X_RIGHT,  Y_BOT+1)]:
        pg.draw.rect(screen, STOPLINE, cell_rect(x,y))

def draw_car_with_id(screen, name, rect, color, cid, font):
    pg.draw.rect(screen, color, rect)
    # маркер направления (точка на «носу»)
    x,y,w,h = rect
    if name == "WE": p = (x+w-3, y+h//2)
    elif name == "EW": p = (x+3,   y+h//2)
    elif name == "NS": p = (x+w//2, y+h-3)
    else:             p = (x+w//2, y+3)
    pg.draw.circle(screen, (15,15,15), p, 2)
    # ID
    label = font.render(str(cid), True, (10,10,10))
    screen.blit(label, (x+2, y+1))

def draw_scene(screen, lanes, t, moved, font_id, font_ui):
    screen.fill(BG)
    # поле дорог
    pg.draw.rect(screen, ASPHALT, (PAD, PAD, L*CELL, L*CELL))
    # центр 2×2
    for (x,y) in [(X_LEFT,Y_TOP),(X_RIGHT,Y_TOP),(X_LEFT,Y_BOT),(X_RIGHT,Y_BOT)]:
        pg.draw.rect(screen, INT_MARK, cell_rect(x,y))

    draw_lane_highlight(screen)
    draw_dashed_centerlines(screen)
    draw_stop_lines(screen)
    draw_grid(screen)

    # машины вне 2×2
    for name, ln, col in (("WE",lanes[0],CAR_EW),("EW",lanes[1],CAR_EW),
                          ("NS",lanes[2],CAR_NS),("SN",lanes[3],CAR_NS)):
        for c in ln.cars:
            if c.stage == 0:
                draw_car_with_id(screen, name, lane_cell_rect(name, c.pos), col, c.cid, font_id)

    # машины внутри 2×2 — рисуем и ID тоже
    for name, ln, col in (("WE",lanes[0],CAR_EW),("EW",lanes[1],CAR_EW),
                          ("NS",lanes[2],CAR_NS),("SN",lanes[3],CAR_NS)):
        a,b = cross_path(name)
        for c in ln.cars:
            if c.stage in (1,2):
                xy = a if c.stage==1 else b
                rect = cell_rect(*xy)
                pg.draw.rect(screen, col, rect)
                # ID поверх
                label = font_id.render(str(c.cid), True, (10,10,10))
                screen.blit(label, (rect[0]+2, rect[1]+1))


    # UI и лампы
    who = light(t)
    txt = f"t={t}  light={who}  moved={moved}  NS={CARS_NS}  EW={CARS_EW}"
    screen.blit(font_ui.render(txt, True, TEXT), (PAD, 6))

    lamp_ns = GREEN if who == 'NS' else RED
    lamp_ew = GREEN if who == 'EW' else RED

    # центр перекрестка в пикселях
    cx = PAD + ((X_LEFT + X_RIGHT + 1) / 2) * CELL
    cy = PAD + ((Y_TOP + Y_BOT + 1) / 2) * CELL
    r = max(6, CELL // 2)  # радиус «лампы»
    d = 2 * CELL  # отступ от центра

    # Горизонтальная пара (влево/вправо от центра) — фаза EW
    pg.draw.circle(screen, lamp_ew, (int(cx - d), int(cy)), r)
    pg.draw.circle(screen, lamp_ew, (int(cx + d), int(cy)), r)

    # Вертикальная пара (вверх/вниз от центра) — фаза NS
    pg.draw.circle(screen, lamp_ns, (int(cx), int(cy - d)), r)
    pg.draw.circle(screen, lamp_ns, (int(cx), int(cy + d)), r)


# ===================== MAIN =====================
def main():
    global GRID_ON
    pg.init()
    screen = pg.display.set_mode(scr_size())
    pg.display.set_caption("CA — соосный перекресток 2×2: строгие фазы, ID и маркеры")
    clock = pg.time.Clock()
    font_id = pg.font.SysFont("consolas", 12)   # для номеров
    font_ui = pg.font.SysFont("consolas", 14)   # для статуса

    lanes = make_world()
    t = 0
    paused = False
    tick = TICK_MS

    while True:
        for e in pg.event.get():
            if e.type == pg.QUIT:
                pg.quit(); sys.exit(0)
            if e.type == pg.KEYDOWN:
                if e.key == pg.K_SPACE: paused = not paused
                elif e.key in (pg.K_PLUS, pg.K_EQUALS):    tick = max(15, tick-10)
                elif e.key in (pg.K_MINUS, pg.K_UNDERSCORE): tick = min(500, tick+10)
                elif e.key == pg.K_r: lanes = make_world(); t = 0
                elif e.key == pg.K_g: GRID_ON = not GRID_ON

        moved = 0
        if not paused:
            moved = one_step(lanes, t)
            t += 1

        draw_scene(screen, lanes, t, moved, font_id, font_ui)
        pg.display.flip()
        clock.tick(1000 // max(1, tick))

if __name__ == "__main__":
    main()
