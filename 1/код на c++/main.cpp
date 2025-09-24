#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <thread>



void fill_grid(std::vector<std::vector<int>>& grid) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, 2);

    for (auto& row : grid) {
        for (auto& val : row) {
            val = distrib(gen);
        }
    }
}

void print_grid(const std::vector<std::vector<int>>& grid) {
    for (const auto& row : grid) {
        for (const auto& val : row) {
            if (val == 0) std::cout << "⬛";      
            else if (val == 1) std::cout << "🟥"; 
            else std::cout << "🟩";               
        }
        std::cout << '\n';
    }
}

bool make_destroy_square(std::vector<std::vector<int>>& grid) {
    const std::vector<std::pair<int, int>> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-1, -1}, {1, 1},{-1, 1}, {1, -1}};
    std::vector<std::pair<int, int>> to_burn;
    bool fire_spread = false;
    
    for (int i = 0; i < grid.size(); i++) {
        for (int j = 0; j < grid[i].size(); j++) {
            if (grid[i][j] == 1) {
                for (const auto& dir : directions) {
                    int new_x = i + dir.first;
                    int new_y = j + dir.second;
                    
                    if (new_x >= 0 && new_x < grid.size() && 
                        new_y >= 0 && new_y < grid[0].size()) {
                        if (grid[new_x][new_y] == 2) {
                            to_burn.emplace_back(new_x, new_y);
                            fire_spread = true;
                        }
                    }
                }
                grid[i][j] = 0;
            }
        }
    }
    
    for (const auto& pos : to_burn) {
        grid[pos.first][pos.second] = 1;
    }
    
    return fire_spread;
}

bool make_destroy_triangle(std::vector<std::vector<int>>& grid) {
    const std::vector<std::pair<int, int>> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    std::vector<std::pair<int, int>> to_burn;
    bool fire_spread = false;
    
    for (int i = 0; i < grid.size(); i++) {
        for (int j = 0; j < grid[i].size(); j++) {
            if (grid[i][j] == 1) {
                for (const auto& dir : directions) {
                    int new_x = i + dir.first;
                    int new_y = j + dir.second;
                    
                    if (new_x >= 0 && new_x < grid.size() && 
                        new_y >= 0 && new_y < grid[0].size()) {
                        if (grid[new_x][new_y] == 2) {
                            to_burn.emplace_back(new_x, new_y);
                            fire_spread = true;
                        }
                    }
                }
                grid[i][j] = 0;
            }
        }
    }
    
    for (const auto& pos : to_burn) {
        grid[pos.first][pos.second] = 1;
    }
    
    return fire_spread;
}

int main(void) {
    int n = 50;
    std::vector<std::vector<int>> grid(n, std::vector<int>(n));
    
    fill_grid(grid);
    
    grid[n/2][n/2] = 1;
    
    int iteration = 0;
    bool fire_continues = true;
    
    while (fire_continues) {
        print_grid(grid);
        fire_continues = make_destroy_square(grid);
        
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
    
    return 0;
}
