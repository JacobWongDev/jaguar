#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include "logger.h"
#include "cuda/cuda_util.h"

/**
 * @brief Prints banner and performs various system checks.
 *
 * @param argc Number of command-line arguments
 * @param argv Array of command-line arguments
 */
bool init() {
    // Fetch banner and print it
    FILE* banner_file = fopen("../src/resources/banner.txt", "r");
    char* line = NULL;
    size_t len = 0;
    ssize_t nread;
    if(banner_file != NULL) {
        while((nread = getline(&line, &len, banner_file)) != -1) {
            fwrite(line, nread, 1, stdout);
        }
        free(line);
        fclose(banner_file);
    } else {
        logger_send("Couldn't open banner file!", ERROR);
        return false;
    }
    // Find and initialize CUDA device
    return cuda_init();
}

/**
 * @brief main function
 *
 * @param argc Number of command-line arguments
 * @param argv Array of command-line arguments
 * @return Status code
 */
int main(int argc, char* argv[]) {
    if(init()) {
        //TODO
        return EXIT_SUCCESS;
    } else {
        return EXIT_FAILURE;
    }
}