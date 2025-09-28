#include <iostream>
#include <thread>
#include <vector>

using namespace std;

struct Ant {
    int x, y;
    int dir;
};

void iteration(vector<vector<int>>& grid, Ant& ant) {
    int n = grid.size();

    if(grid[ant.y][ant.x] == 0){
        grid[ant.y][ant.x] = 1;
        ant.dir = (ant.dir+1)%4;
    } else {
        grid[ant.y][ant.x] = 0;
        ant.dir = (ant.dir+3)%4;
    }

    if (ant.dir == 0) 
        ant.y = (ant.y - 1 + n) % n;
    else if (ant.dir == 1) 
        ant.x = (ant.x + 1) % n;
    else if (ant.dir == 2) 
        ant.y = (ant.y + 1) % n;
    else if (ant.dir == 3) 
        ant.x = (ant.x - 1 + n) % n;
}

void printGrid(const vector<vector<int>>& grid, const Ant& ant) {
    int n = grid.size();
    for (int y = 0; y < n; y++) {
        for (int x = 0; x < n; x++) {
            if (x == ant.x && y == ant.y) cout << "🟥";
            else cout << (grid[y][x] ? "⬛️" : "⬜️");
        }
        cout << "\n";
    }
    cout << string(20, '-') << "\n";
}

int main() {
    int n = 50;
    vector<vector<int>> grid(n, vector<int>(n, 0));

    Ant ant = {n/2, n/2, 0};

    for (int i = 0; i < 10000; i++) {
        iteration(grid, ant);
        printGrid(grid, ant);
        this_thread::sleep_for(chrono::seconds(1));
    }

    return 0;
}