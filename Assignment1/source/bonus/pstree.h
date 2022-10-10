#include <dirent.h>
#include <linux/unistd.h>
#include <stdio.h>
#include <string.h>

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

int getPPid(char* filename);
void setPid_PPid();
