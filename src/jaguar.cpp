#include <unistd.h>
#include "spdlog/spdlog.h"
#include "util/cuda_util.h"
#include "cosq.h"
#include <iostream>

#define MAX_BIT_RATE 10
#define MAX_TRAINING_SIZE (1 << 25)
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
        spdlog::error("Could not open banner file!");
        return false;
    }
    // Find and initialize CUDA device
    return cuda_init();
}

bool isTrainingPow2(unsigned int x) {
  return ((x & (x - 1)) == 0);
}

bool verify_args(unsigned int* bit_rate, unsigned int* training_size) {
    if(*training_size == 0 || *training_size >= MAX_TRAINING_SIZE || !isTrainingPow2(*training_size))
        return false;
    if(*bit_rate == 0 || *bit_rate >= MAX_BIT_RATE)
        return false;
    return true;
}

double* read_training_seq(char* file_name, unsigned int* training_size) {
    FILE* training_seq_file = fopen(file_name, "r");
    double* training_sequence = (double*) malloc(*training_size * sizeof(double));
    if(training_seq_file != NULL) {
        size_t count = fread(training_sequence, sizeof(double), *training_size, training_seq_file);
        if(count != *training_size) {
            spdlog::error("Could not read entire training sequence!");
            free(training_sequence);
            exit(EXIT_FAILURE);
        }
    } else {
        spdlog::error("Could not open training sequence file!");
        fclose(training_seq_file);
        return NULL;
    }
    fclose(training_seq_file);
    return training_sequence;
}

void write_quantizer(double* quantizer, unsigned int* bit_rate) {
    FILE* quantizer_file = fopen("quantizer", "w");
    size_t count = fwrite(quantizer, sizeof(double), (1 << *bit_rate), quantizer_file);
    if(count != (1 << *bit_rate)) {
        spdlog::error("Could not write entire quantizer to file!");
    }
    fclose(quantizer_file);
}

/**
 * @brief main function
 *
 * @param argc Number of command-line arguments
 * @param argv Array of command-line arguments
 * @return Status code
 */
int main(int argc, char** argv) {
    int opt;
    bool bit_rate_set = false;
    bool training_size_set = false;
    bool training_file_set = false;
    unsigned int bit_rate, training_size;
    char* training_sequence_file;
    // Parse CLI arguments
    while((opt = getopt(argc, argv, "b:t:f:")) != -1) {
        switch(opt) {
            case 'b':
                bit_rate = std::strtoul(optarg, nullptr, 10);
                bit_rate_set = true;
            break;
            case 't':
                training_size = std::strtoul(optarg, nullptr, 10);
                training_size_set = true;
            break;
            case 'f':
                training_sequence_file = optarg;
                training_file_set = true;
            break;
            default:
                spdlog::error("Unknown command line option.");
                spdlog::error("Please use -b to specify bit rate, -t for training size and -f for training sequence file.");
                return EXIT_FAILURE;
        }
    }
    // Check if all args are provided
    if(!bit_rate_set || !training_size_set || !training_file_set) {
        spdlog::error("Arguments missing! -b, -t, or -f.");
        spdlog::error("Please use -b to specify bit rate, -t for training size and -f for training sequence file.");
        return EXIT_FAILURE;
    }
    // Verify that provided args are valid
    if(!verify_args(&bit_rate, &training_size)) {
        spdlog::error("Bit rate or training size is invalid!");
        spdlog::error("Bit rate and training size must be non-zero.");
        spdlog::error("Bit rate <= 10 and training size <= 2^25");
        spdlog::error("training size must be a power of 2");
        return EXIT_FAILURE;
    }
    if(init()) {
        // Prepare training sequence
        double* q_points = (double*) malloc(sizeof(double) * (1 << bit_rate));
        double* training_sequence = read_training_seq(training_sequence_file, &training_size);
        spdlog::info("Training COSQ with bit rate {:d} and training size {:d}", bit_rate, training_size);
        COSQ cosq(training_sequence, &training_size, &bit_rate);
        cosq.train(q_points);
        // Write results
        write_quantizer(q_points, &bit_rate);
        free(training_sequence);
        free(q_points);
        return EXIT_SUCCESS;
    } else {
        return EXIT_FAILURE;
    }
}