#include <linux/unistd.h>
#include <stdio.h>
#include <string.h>

#define PATH "/proc"
#define MAX_PROCESS_INFO 5000
#define BUFFSIZE_INFO 100

typedef struct process_info
{
    int pid;
    int ppid;
    char name[100];
} PINFO;

typedef struct proc
{
    struct proc* next;
    struct proc* parent;
    struct child* children;
    int pid;
    char pName[50];
} PROC;
typedef struct child
{
    const PROC* proc;
    struct child* next;
} CHILD;

PINFO processInfos[MAX_PROCESS_INFO];
int number_process;

int getPPid(char* filename)
{
    int ppid = -100;
    char* right = NULL;
    FILE* fp = fopen(filename, "r");
    char info[BUFFSIZE_INFO + 1];
    info[BUFFSIZE_INFO] = '\0';

    if (fp == NULL)
    {
        fprintf(stderr, "open file %s error!\n", filename);
        return -1;
    }

    if (fgets(info, BUFFSIZE_INFO, fp) == NULL)
    {
        puts("fgets error!");
        exit(0);
    }
    right = strrchr(info, ')');
    if (right == NULL)
    {
        printf("not find )\n");
    }
    right += 3;

    sscanf(right, "%d", &ppid);

    return ppid;
}

void setPid_PPid()
{
    DIR* dir_ptr;
    struct dirent* direntp;
    int pid;
    int ppid;
    char process_path[51] = "/proc/";
    char stat[6] = "/stat";
    char pidStr[20];

    dir_ptr = opendir(PATH);

    if (dir_ptr == NULL)
    {
        fprintf(stderr, "can not open /proc\n");
        exit(0);
    }

    while (direntp = readdir(dir_ptr))
    {
        pid = atoi(direntp->d_name);
        if (pid != 0)
        {
            processInfos[number_process].pid = pid;

            sprintf(pidStr, "%d", pid);
            strcat(process_path, pidStr);
            strcat(process_path, stat);

            int ppid = getPPid(process_path);
            if (ppid != -1)
                processInfos[number_process++].ppid = ppid;
            else
                number_process++;

            process_path[6] = 0;
        }
    }
}

PROC* add_proc(const char* name, int pid, int ppid)
{
    PROC* this = find_proc(pid);

    if (this != NULL)
    {
        rename_proc(this, name);
    }
    else
    {
        this = new_proc(name, pid);
        if (this == NULL)
        {
            printf("new_proc error!\n");
        }
        this->next = list;
        list = this;
    }

    if (ppid == -1)
    {
        return this;
    }

    PROC* parent = find_proc(ppid);
    if (parent == NULL)
    {
        parent = new_proc("?", ppid);
        parent->next = list;
        list = parent;
    }

    this->parent = parent;
    add_child(parent, this);

    return this;
}

int main()
{
    setPid_PPid();
    for (int i = 0; i < number_process; i++)
    {
        printf("%d %d\n", processInfos[i].pid, processInfos[i].ppid);
    }
    return 0;
}