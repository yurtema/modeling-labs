#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>
#include <random>
#include <chrono>
#include <iomanip>
#include <fstream>

// ------------------------- Петербургская игра ---------------------------

enum coin { eagle = 0, tails = 1 };

coin coin_toss()
{
    return coin(std::rand() % 2);
}

// Петербургская игра: выигрыш 1, 2, 4, 8, ...
int start_game()
{
    int value = 1;
    coin toss;
    do {
        toss = coin_toss();
        if (toss == eagle)
            value <<= 1; // умножаем на 2
    } while (toss == eagle);
    // если сразу решка -> 1 = 2^0
    return value;
}

// --------------------- Модель «депозит + много игр» ---------------------

// Полная запись одной сессии (нужна только для красивого графика)
struct GameRun {
    int  initialDeposit{};
    int  finalCapital{};
    int  steps{};          // сколько партий сыграли
    bool win{};            // true, если вышли в плюс относительно начального депозита
    bool hitLimit{};       // true, если остановились из-за лимита шагов
    std::vector<int> capitalHistory; // динамика капитала по шагам
};

// простой исход сессии без истории — используем в статистике
struct SimpleOutcome {
    int  finalCapital{};
    int  steps{};
    bool win{};
};

// большой, но конечный лимит шагов, чисто как страховка
const long long HARD_MAX_STEPS = 50'0000'000LL;

// -------- 1) медленный, но подробный симулятор (с историей и лимитом) ----

// стартуем с депозита D, ставка cost_game.
// Играем, пока есть деньги на ставку и мы не вышли "в плюс":
//   - если капитал > initialDeposit  -> победа;
//   - если капитал < cost_game       -> проигрыш.
GameRun simulate_deposit_game_detailed(int initialDeposit, int cost_game)
{
    GameRun result;
    result.initialDeposit = initialDeposit;
    int capital = initialDeposit;

    result.capitalHistory.clear();
    result.capitalHistory.push_back(capital);

    long long step = 0;
    bool limit = false;

    while (capital >= cost_game && capital <= initialDeposit) {
        if (step >= HARD_MAX_STEPS) {
            limit = true;
            break;
        }

        int winnings = start_game();           // 1, 2, 4, ...
        capital += winnings - cost_game;       // чистый результат партии
        if (capital < 0) capital = 0;          // на всякий случай

        ++step;
        result.capitalHistory.push_back(capital);
    }

    result.finalCapital = capital;
    result.steps = static_cast<int>(step);
    result.win = (capital > initialDeposit);
    result.hitLimit = limit;
    return result;
}

// -------- 2) быстрый симулятор для статистики (БЕЗ лимита и истории) ----

SimpleOutcome simulate_deposit_game_simple(int initialDeposit, int cost_game)
{
    SimpleOutcome out{};
    int capital = initialDeposit;
    int steps = 0;

    while (capital > cost_game && capital <= initialDeposit) {
        ++steps;
        int winnings = start_game();
        capital += winnings - cost_game;
        if (capital < 0) capital = 0;
    }

    out.finalCapital = capital;
    out.steps = steps;
    out.win = (capital > initialDeposit);
    return out;
}

// --------------------------- CSV-сохранение -----------------------------

// динамика капитала одной «сессии»
void save_single_game_csv(int deposit, const GameRun& run)
{
    std::string fname = "deposit_" + std::to_string(deposit) + "_single.csv";
    std::ofstream out(fname);
    if (!out) {
        std::cerr << "Не удалось открыть файл " << fname << " для записи.\n";
        return;
    }
    out << "step,capital\n";
    for (size_t i = 0; i < run.capitalHistory.size(); ++i) {
        out << i << "," << run.capitalHistory[i] << "\n";
    }
}

// краткая инфа про одну сессию (для 20 штук)
struct GameSummary {
    int  index{};
    int  steps{};
    bool win{};
};

// 20 сессий для диаграммы «кол-во игр до выигрыша/проигрыша»
std::vector<GameSummary> simulate_twenty_games(int deposit, int cost_game)
{
    std::vector<GameSummary> games;
    games.reserve(20);

    for (int i = 0; i < 20; ++i) {
        SimpleOutcome out = simulate_deposit_game_simple(deposit, cost_game);
        games.push_back({ i + 1, out.steps, out.win });
    }
    return games;
}

void save_twenty_games_csv(int deposit,
    const std::vector<GameSummary>& games)
{
    std::string fname = "deposit_" + std::to_string(deposit) + "_20games.csv";
    std::ofstream out(fname);
    if (!out) {
        std::cerr << "Не удалось открыть файл " << fname << " для записи.\n";
        return;
    }
    out << "game,steps,win\n";
    for (const auto& g : games) {
        out << g.index << "," << g.steps << "," << (g.win ? 1 : 0) << "\n";
    }
}

// 1000 сессий — статистика по депозиту
struct DepositStats {
    int    deposit{};
    int    totalGames{};
    int    wins{};
    int    losses{};
    double meanProfit{}; // средний выигрыш относительно депозита
};

DepositStats thousand_games_stats(int deposit, int cost_game)
{
    DepositStats s;
    s.deposit = deposit;
    s.totalGames = 1000;

    long long totalProfit = 0;

    for (int i = 0; i < s.totalGames; ++i) {
        SimpleOutcome out = simulate_deposit_game_simple(deposit, cost_game);
        int profit = out.finalCapital - deposit;
        totalProfit += profit;

        if (out.win)
            ++s.wins;
        else
            ++s.losses;
    }

    s.meanProfit = static_cast<double>(totalProfit) / s.totalGames;
    return s;
}

// ----------------- Кинетический метод Монте-Карло (по слайду) -----------

struct KMCSnapshot {
    double t;
    int A;
    int B;
};

std::vector<KMCSnapshot> kinetic_monte_carlo_dt(double k1, double k2,
    int A0, int B0,
    double dt, double t_max)
{
    std::vector<KMCSnapshot> traj;

    std::mt19937 rng(static_cast<unsigned>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    int A = A0;
    int B = B0;
    const int N = A + B; // должно оставаться константой

    double t = 0.0;
    int nSteps = static_cast<int>(std::floor(t_max / dt));

    traj.reserve(nSteps + 1);
    traj.push_back({ t, A, B });

    for (int step = 0; step < nSteps; ++step) {
        for (int n = 0; n < N; ++n) {
            double u = uni(rng);
            bool chooseA = (u < static_cast<double>(A) / N);

            if (chooseA) {
                if (A > 0) {
                    double u2 = uni(rng);
                    if (u2 < k1 * dt) {
                        --A; ++B; // A -> B
                    }
                }
            }
            else {
                if (B > 0) {
                    double u2 = uni(rng);
                    if (u2 < k2 * dt) {
                        --B; ++A; // B -> A
                    }
                }
            }
        }
        t += dt;
        traj.push_back({ t, A, B });
    }

    return traj;
}

void save_kmc_traj_csv(const std::vector<KMCSnapshot>& traj,
    const std::string& fname)
{
    std::ofstream out(fname);
    if (!out) {
        std::cerr << "Не удалось открыть файл " << fname << " для записи.\n";
        return;
    }
    out << "t,A,B\n";
    for (const auto& p : traj) {
        out << std::fixed << std::setprecision(6)
            << p.t << "," << p.A << "," << p.B << "\n";
    }
}

// ------------------------------ main -----------------------------------

int main()
{
    setlocale(0, "rus");
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    const int cost_game = 10; // Вариант 3: стоимость игры 10 у.е.
    std::vector<int> deposits = { 100, 1000, 100000, 1000000 };

    // общий файл со статистикой по 1000 играм
    std::ofstream statsFile("deposit_stats.csv");
    if (!statsFile) {
        std::cerr << "Не удалось открыть deposit_stats.csv для записи.\n";
        return 1;
    }
    statsFile << "deposit,totalGames,wins,losses,meanProfit\n";

    for (int dep : deposits) {
        std::cout << "Моделируем депозит " << dep << "...\n";

        // 1) ОДНА подробная сессия с историей – для графика
        GameRun one = simulate_deposit_game_detailed(dep, cost_game);
        save_single_game_csv(dep, one);

        // 2) 20 сессий — шаги до выигрыша/проигрыша (быстрый симулятор)
        auto games20 = simulate_twenty_games(dep, cost_game);
        save_twenty_games_csv(dep, games20);

        // 3) статистика по 1000 сессиям (быстрый симулятор)
        DepositStats st = thousand_games_stats(dep, cost_game);
        statsFile << st.deposit << ","
            << st.totalGames << ","
            << st.wins << ","
            << st.losses << ","
            << std::fixed << std::setprecision(6)
            << st.meanProfit << "\n";
    }

    // -------- часть 2: кинетический метод Монте-Карло ---------

    const int A0 = 500;
    const int B0 = 200;
    const double k1 = 0.6;  // вариант 3
    const double k2 = 0.9;
    const double dt = 0.02; // как в примере
    const double t_max = 4.0;

    auto traj = kinetic_monte_carlo_dt(k1, k2, A0, B0, dt, t_max);
    save_kmc_traj_csv(traj, "kmc_traj.csv");

    std::cout << "Моделирование завершено, CSV-файлы готовы.\n";
    return 0;
}
