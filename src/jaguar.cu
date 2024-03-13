#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <getopt.h>
#include "util/logger.hpp"
#include "util/cuda_util.hpp"
#include "util/pgm_util.hpp"
#include "training/simple_training.cuh"
#include "transmission/transmission.cuh"
#include "cuda/bit_allocations.h"

enum MODE {
    NONE = 0,
    TRAINING = 1,
    CHANNEL_TRANSMISSION = 2
};

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
        line = NULL;
        fclose(banner_file);
    } else {
        logger_send("Couldn't open banner file!", log_level::ERROR);
        return false;
    }
    // Find and initialize CUDA device
    return cuda_init();
}

int parse_args(int argc, char** argv, char** quantizer_name, char** dir_name, char** channel_name) {
    int c = 0;
    enum MODE mode = NONE;
    while((c = getopt(argc, argv, "c:d:q:CT")) != -1) {
        switch(c) {
            case 'T':
                if(mode == NONE)
                    mode = TRAINING;
                else
                    logger_send("Cannot be in two modes! Choose only one.",  log_level::ERROR);
            break;
            case 'C':
                if(mode == NONE)
                    mode = CHANNEL_TRANSMISSION;
                else
                    logger_send("Cannot be in two modes! Choose only one.",  log_level::ERROR);
            break;
            case 'c':
                *channel_name = optarg;
            break;
            case 'd':
                *dir_name = optarg;
            break;
            case 'q':
                *quantizer_name = optarg;
            break;
            // case '?':
            //     //TODO
            // default:
            //     //TODO
        }
    }
    if(mode == NONE) {
        logger_send("CLI Arguments: Need to select at least 1 mode!",  log_level::ERROR);
    }
    return mode;
}

/**
 * @brief main function
 *
 * The CLI arguments dictate the behaviour of the program. There are two program modes:
 *  - Training mode:
 *    - Format: ./cosq -T -d <dir_name> -c <channel_name>;
 *    - <dir_name> is the directory containing all training images in .pbm format. They must all have the same dimensions!
 *    - <channel_name> is the name of the target channel. The algorithm will optimize the quantizer for the particular channel.
 *    - After the program terminates, assuming no errors, a file called quantizer_<channel_name>_timestamp.txt should be generated.
 *
 *  - Channel Transmission mode:
 *    - Format: ./cosq -C -d <dir_name> -q <quantizer> -c <channel_name>
 *    - <quantizer> is a file (plain text) containing the N different quantizers generated during training.
 *    - <channel_name> is the name of the target channel. The algorithm will optimize the quantizer for the particular channel.
 *
 * @param argc Number of command-line arguments
 * @param argv Array of command-line arguments
 * @return Status code
 */
int main(int argc, char** argv) {
    int mode = 0;
    char* quantizer_file = NULL;
    char* dir_name = NULL;
    char* channel_name = NULL;
    if(init() && (mode = parse_args(argc, argv, &quantizer_file, &dir_name, &channel_name))) {
        switch(mode) {
            case TRAINING:
                train(__76_bit);
            break;
            case CHANNEL_TRANSMISSION:
                transmit_over_channel(__76_bit, dir_name, quantizer_file, channel_name);
            break;
            case NONE:
                return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    } else {
        return EXIT_FAILURE;
    }
}