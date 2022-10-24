#include <curses.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>
#include <cstddef>
#include <cstdlib>

#define ROW 10
#define COLUMN 50
#define GAME_TICK 5
#define FPS_MAX 100

struct Node {
    int x, y;
    int dir = 0;
    Node(int _x, int _y) : x(_x), y(_y){};
    Node(){};
} frog;

struct Wood {
    int width;
    Node state;
} woods[ROW + 1];

struct Frame {
    char map[ROW + 10][COLUMN];
} frame;

char map[ROW + 10][COLUMN];
pthread_t t_woods;
pthread_t t_frog;
pthread_t t_render[ROW];
pthread_t t_backend;
int GAME_FLAG = 0;

pthread_mutex_t cursor_lock;

// Determine a keyboard is hit or not. If yes, return 1. If not, return 0.
int kbhit(void) {
    struct termios oldt, newt;
    int ch;
    int oldf;

    tcgetattr(STDIN_FILENO, &oldt);

    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);

    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    oldf = fcntl(STDIN_FILENO, F_GETFL, 0);

    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

    ch = getchar();

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    fcntl(STDIN_FILENO, F_SETFL, oldf);

    if (ch != EOF) {
        ungetc(ch, stdin);
        return 1;
    }
    return 0;
}

void* logs_move(void* t) {
    while (GAME_FLAG == 0) {
        /*  Move the logs  */
        
        if (kbhit()) {
            char ch = getchar();
            if (ch == 'w' || ch == 'W')
                frog.y -= 1;
            else if (ch == 's' || ch == 'S') {
                frog.y += 1;
            }
            else if (ch == 'a' || ch == 'A') {
                frog.x -= 1;
            }
            else if (ch == 'd' || ch == 'D') {
                frog.x += 1;
            }
            
        }
        
        /*  Check keyboard hits, to change frog's position or quit the game. */
        frog.dir = woods[frog.y].state.dir;
        /*  Check game's status  */

        /*  Print the map on the screen  */
        int pos_x = frog.x, pos_y = frog.y;
        if (pos_y == 0) {
            GAME_FLAG = 1;
            break;
        }
        if (pos_x < 0 || pos_x >= COLUMN - 1) {
            GAME_FLAG = -1;
            break;
        }
        if (pos_x >= 0 && pos_x < COLUMN - 1 && pos_y != ROW && 
            abs(pos_x - woods[pos_y].state.x) > woods[pos_y].width) {
            GAME_FLAG = -1;
            break;
        }
        usleep(1000000 / GAME_TICK);
    }
    return nullptr;
}

void* frog_move(void* args) {
    while (GAME_FLAG == 0) {
        if (GAME_FLAG != 0) {
            break;
        }
        frog.x = frog.x + frog.dir;
        usleep(1000000 / GAME_TICK);
    }
    return nullptr;
}

void* wood_move(void* wood) {
    while (GAME_FLAG == 0) {
        if (GAME_FLAG != 0) {
            break;
        }
        for (int i = 1; i < ROW; ++i) {
            // printf("Row: %d Dir: %d\n", i, woods[i].state.dir);
            woods[i].state.x =
                (woods[i].state.x + woods[i].state.dir + COLUMN - 1) %
                (COLUMN - 1);
        }
        usleep(1000000 / GAME_TICK);
    }
    return nullptr;
}

void* render_row(void* row) {
    int row_num = (int)(*((int*)row));
    while (GAME_FLAG == 0) {
        if (GAME_FLAG != 0) {
            break;
        }
        if (row_num == 0 || row_num == ROW) {
            for (int i = 0; i < COLUMN - 1; ++i)
                map[row_num][i] = '|';
        }
        if (row_num > 0 && row_num < ROW) {
            int pos_x = woods[row_num].state.x,
                pos_width = woods[row_num].width;

            int head = pos_x - pos_width;
            int tail = pos_x + pos_width;
            if (head < 0) {
                for (int i = 0; i <= tail; ++i) {
                    map[row_num][i] = '=';
                }
                for (int i = tail + 1; i < COLUMN - 1 + head; ++i) {
                    map[row_num][i] = ' ';
                }
                for (int i = COLUMN - 1 + head; i < COLUMN - 1; ++i) {
                    map[row_num][i] = '=';
                }
            } else if (tail >= COLUMN - 1) {
                for (int i = head; i < COLUMN - 1; ++i) {
                    map[row_num][i] = '=';
                }
                for (int i = tail - COLUMN + 1; i < head; ++i) {
                    map[row_num][i] = ' ';
                }
                for (int i = 0; i < tail - COLUMN + 1; ++i) {
                    map[row_num][i] = '=';
                }
            } else {
                for (int i = 0; i < head; i++)
                    map[row_num][i] = ' ';
                for (int i = head; i <= tail; i++)
                    map[row_num][i] = '=';
                for (int i = tail + 1; i < COLUMN - 1; i++)
                    map[row_num][i] = ' ';
            }
        }

        if (frog.y == row_num) {
            int pos_frog = frog.x;
            map[row_num][pos_frog] = '0';
        }
        pthread_mutex_lock(&cursor_lock);
        printf("\033[?25l\033[%d;1H\033[K", row_num + 1);
        // printf("%d", row_num);
        printf("%s", map[row_num]);
        pthread_mutex_unlock(&cursor_lock);

        usleep(1000000 / FPS_MAX);
    }
    return nullptr;
}

void init() {
    int dir = (rand() % 2) * 2 - 1;
    woods[1].width = rand() % 5 + 4;
    woods[1].state.dir = dir;
    woods[1].state.x = rand() % (COLUMN - 1);
    woods[1].state.y = 1;
    for (int i = 2; i < ROW; ++i) {
        woods[i].width = rand() % 5 + 4;
        woods[i].state.dir = -woods[i - 1].state.dir;
        woods[i].state.x = rand() % (COLUMN - 1);
        woods[i].state.y = i;
    }
    pthread_mutex_init(&cursor_lock, NULL);
    printf("\033[2J");
    return;
}

void endGame() {
    printf("\033[2J");
    switch (GAME_FLAG) {
        case 1:
            printf("\033[2J\033[H");
            printf("You win the game!!");
            break;
        case -1:
            printf("\033[2J\033[H");
            printf("You lose the game!!");
            break;
    }
    return;
}

int main(int argc, char* argv[]) {
    // Initialize the river map and frog's starting position
    memset(map, 0, sizeof(map));
    srand(int(time(0)));
    for (int i = 1; i < ROW; ++i) {
        for (int j = 0; j < COLUMN - 1; ++j)
            map[i][j] = ' ';
    }

    for (int j = 0; j < COLUMN - 1; ++j)
        map[ROW][j] = map[0][j] = '|';

    for (int j = 0; j < COLUMN - 1; ++j)
        map[0][j] = map[0][j] = '|';

    frog = Node((COLUMN - 1) / 2, ROW);
    frog.dir = 0;
    map[frog.y][frog.x] = '0';

    init();

    // Print the map into screen

    /*  Create pthreads for wood move and frog control.  */

    int error_frog = pthread_create(&t_frog, NULL, frog_move, NULL);
    if (error_frog != 0) {
        printf("\nThread can't be created :[%s]", strerror(error_frog));
    }

    // Create wood threads row1-9

    int error_wood = pthread_create(&t_woods, NULL, wood_move, &woods);
    if (error_wood != 0) {
        printf("\nThread can't be created :[%s]", strerror(error_wood));
    }

    int error_backend = pthread_create(&t_backend, NULL, logs_move, NULL);
    if (error_backend != 0) {
        printf("\nThread can't be created :[%s]", strerror(error_backend));
    }
    // Create render threads row0-10
    for (int i = 0; i <= ROW; ++i) {
        int* row_num = new int();
        *row_num = i;
        int error_render =
            pthread_create(&t_render[i], NULL, render_row, row_num);
        if (error_render != 0) {
            printf("\nThread can't be created :[%s]", strerror(error_render));
        }
    }

    pthread_join(t_backend, NULL);
    pthread_join(t_woods, NULL);
    pthread_join(t_frog, NULL);
    for (int i = 0; i < ROW; ++i)
        pthread_join(t_render[i], NULL);
    /*  Display the output for user: win, lose or quit.  */
    endGame();
    return 0;
}
