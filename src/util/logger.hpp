#include <iostream>

enum log_level {
    DEBUG,
    INFO,
    ERROR
};

void logger_send(const char* message, enum log_level level);
