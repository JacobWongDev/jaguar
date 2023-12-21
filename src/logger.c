#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "logger.h"

time_t current_time;

void logger_send(const char* message, enum log_level level) {
    time(&current_time);
    struct tm* t = localtime(&current_time);
    char buffer[100];
    strftime(buffer, 100, "%a %b %d %H:%M:%S %Y", t);
    switch(level) {
        case DEBUG:
            fprintf(stdout, "(%s) [DEBUG] %s\n", buffer, message);
        break;
        case ERROR:
            fprintf(stderr, "(%s) [ERROR] %s\n", buffer, message);
        break;
        case INFO:
            fprintf(stdout, "(%s) [INFO] %s\n", buffer, message);
        break;
        default:
            fprintf(stderr, "(%s) [ERROR] %s\n", buffer, "Unsupported Logging level used!");
        break;
    }
}