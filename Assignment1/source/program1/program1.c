#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char* argv[])
{
    /* fork a child process */
    pid_t pid = fork();

    if (pid == -1)
    {
        perror("ERROR: fork error.\n");
        exit(1);
    }

    else
    {
        if (pid == 0)
        {
            printf("I'm the Child Process, my pid = %d\n", getpid());
            printf("Child process start to execute test program:\n");

            /* execute test program */
            char* arg[argc];

            for (int i = 0; i < argc - 1; ++i)
            {
                arg[i] = argv[i + 1];
            }
            arg[argc - 1] = NULL;
            execve(arg[0], arg, NULL);

            exit(SIGCHLD);
        }

        else
        {
            printf("I'm the Parent Process, my pid = %d\n", getpid());

            /* wait for child process terminates */
            int status = 0;
            waitpid(pid, &status, WUNTRACED);

            /* check child process'  termination status */
            char sig[20] = "";
            if (WIFEXITED(status))
            {
                printf("Normal termination with EXIT STATUS = %d\n",
                       WEXITSTATUS(status));
            }
            else
            {
                int signal_result = WIFSIGNALED(status);
                if (signal_result)
                {
                    switch (status)
                    {
                    case SIGABRT:
                        strcpy(sig, "SIGABRT");
                        break;
                    case SIGALRM:
                        strcpy(sig, "SIGALRM");
                        break;
                    case SIGBUS:
                        strcpy(sig, "SIGBUS");
                        break;
                    case SIGCONT:
                        strcpy(sig, "SIGCONT");
                        break;
                    case SIGCHLD:
                        strcpy(sig, "SIGCHLD");
                        break;
                    case SIGFPE:
                        strcpy(sig, "SIGFPE");
                        break;
                    case SIGHUP:
                        strcpy(sig, "SIGHUP");
                        break;
                    case SIGILL:
                        strcpy(sig, "SIGILL");
                        break;
                    case SIGINT:
                        strcpy(sig, "SIGINT");
                        break;
                    case SIGKILL:
                        strcpy(sig, "SIGKILL");
                        break;
                    case SIGPIPE:
                        strcpy(sig, "SIGPIPE");
                        break;
                    case SIGQUIT:
                        strcpy(sig, "SIGQUIT");
                        break;
                    case SIGSEGV:
                        strcpy(sig, "SIGSEGV");
                        break;
                    case SIGTSTP:
                        strcpy(sig, "SIGTSTPT");
                        break;
                    case SIGTERM:
                        strcpy(sig, "SIGTERM");
                        break;
                    case SIGTRAP:
                        strcpy(sig, "SIGTRAP");
                        break;
                    default:
                        break;
                    }
                    printf("Parent process receives %s signal\n", sig);
                    printf("CHILD EXECUTION FAILED: %d\n", WTERMSIG(status));
                }
                else if (WIFSTOPPED(status))
                {
                    printf("CHILD PROCESS STOPPED: %d\n", WSTOPSIG(status));
                }
                else
                {
                    printf("CHILD PROCESS CONTINUED\n");
                }
                exit(0);
            }
        }

        return 0;
    }
}