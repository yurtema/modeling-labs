import pygame, random
pygame.init()

GRID_SIZE = 8
COLS, ROWS = 140, 90
W, H = COLS*GRID_SIZE, ROWS*GRID_SIZE
STEPS_PER_FRAME = 20

WRAP = True             # обёртка по краям (тор). Нажми W, чтобы переключить.
steps = 0               # счётчик шагов

screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("Langton's Ant (pygame)")
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 20)

grid = [[0]*COLS for _ in range(ROWS)]
ant_x, ant_y = COLS//2, ROWS//2
direction = 0
paused = False

def wrap_or_clip(x, y):
    if WRAP:
        return x % COLS, y % ROWS
    # без обёртки — останавливаемся у границы
    x = 0 if x < 0 else (COLS-1 if x >= COLS else x)
    y = 0 if y < 0 else (ROWS-1 if y >= ROWS else y)
    return x, y

def draw_cell(x, y):
    color = (0,0,0) if grid[y][x] else (255,255,255)
    pygame.draw.rect(screen, color, (x*GRID_SIZE, y*GRID_SIZE, GRID_SIZE, GRID_SIZE))

def draw_ant():
    pygame.draw.rect(screen, (255,0,0), (ant_x*GRID_SIZE, ant_y*GRID_SIZE, GRID_SIZE, GRID_SIZE))

def reset_grid():
    global grid, ant_x, ant_y, direction, steps
    grid = [[0]*COLS for _ in range(ROWS)]
    ant_x, ant_y = COLS//2, ROWS//2
    direction = 0
    steps = 0

def random_start():
    global ant_x, ant_y, direction, steps
    ant_x, ant_y = random.randrange(COLS), random.randrange(ROWS)
    direction = random.randrange(4)
    steps = 0

def flip_random_cell():
    x, y = random.randrange(COLS), random.randrange(ROWS)
    grid[y][x] = 1 - grid[y][x]

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                paused = not paused
            elif event.key == pygame.K_r:
                reset_grid()
            elif event.key == pygame.K_n:
                random_start()
            elif event.key == pygame.K_b:
                flip_random_cell()     # ломаем симметрию единичным «битом»
            elif event.key == pygame.K_w:
                WRAP = not WRAP        # тор ↔ твёрдая граница
            elif event.key == pygame.K_UP:
                STEPS_PER_FRAME = min(2000, STEPS_PER_FRAME + 10)
            elif event.key == pygame.K_DOWN:
                STEPS_PER_FRAME = max(1, STEPS_PER_FRAME - 10)

    if not paused:
        for _ in range(STEPS_PER_FRAME):
            cell = grid[ant_y][ant_x]
            # поворот
            direction = (direction + 1) % 4 if cell == 0 else (direction - 1) % 4
            # инверсия
            grid[ant_y][ant_x] = 1 - cell
            # шаг
            if direction == 0:
                ant_y -= 1
            elif direction == 1:
                ant_x += 1
            elif direction == 2:
                ant_y += 1
            else:
                ant_x -= 1
            ant_x, ant_y = wrap_or_clip(ant_x, ant_y)
            steps += 1

    # перерисовка поля
    for y in range(ROWS):
        for x in range(COLS):
            draw_cell(x, y)
    draw_ant()

    # HUD: шаги, скорость, режим границ
    hud = f"steps: {steps:,}   spf: {STEPS_PER_FRAME}   wrap: {WRAP}   {'paused' if paused else ''}"
    text_surf = font.render(hud, True, (0,0,0))
    # подложка чтобы читать на любом фоне
    pygame.draw.rect(screen, (220,220,220), (6,6, text_surf.get_width()+8, text_surf.get_height()+6))
    screen.blit(text_surf, (10, 10))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
