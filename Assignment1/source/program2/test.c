#include <signal.h>
#include <stdio.h>
#include <unistd.h>

int main(int argc, char* argv[])
{
    int i = 0;
    freopen("1.txt", "w", stdout);
    printf("--------USER PROGRAM--------\n");
    fclose(stdout);
    //	alarm(2);
    raise(SIGBUS);
    sleep(5);
    printf("user process success!!\n");
    printf("--------USER PROGRAM--------\n");
    return 100;
}
