#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif
    enum log_level {
        DEBUG,
        INFO,
        ERROR
    };

    void logger_send(const char* message, enum log_level level);
#ifdef __cplusplus
}
#endif