#include "spdlog/spdlog.h"
#include "util/cuda_util.h"
#include "cosq.h"
#include <random>

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

/**
 * Return an array of size TRAINING_SIZE containing values distributed according to N(0,1)
*/
double* generate_normal_sequence(unsigned int* training_size) {
    double* normal_sequence = (double*) malloc(*training_size * sizeof(double));
    if(normal_sequence == nullptr) {
        spdlog::error("Memory Allocation error: Failed to allocate memory for normal_sequence!");
        return nullptr;
    }
    std::default_random_engine rng;
    rng.seed(31);
    std::normal_distribution<double> distribution(0, 1);
    for(int i = 0; i < *training_size; i++) {
        normal_sequence[i] = distribution(rng);
    }
    return normal_sequence;
}

/**
 * @brief main function
 *
 * @param argc Number of command-line arguments
 * @param argv Array of command-line arguments
 * @return Status code
 */
int main(int argc, char** argv) {
    if(init()) {
        // Prepare training sequence
        double* q_points = NULL;
        unsigned int bit_rate = 6, training_size = 1 << 20;
        double* training_sequence = generate_normal_sequence(&training_size);
        // Run COSQ algorithm
        q_points = COSQ::train(training_sequence, &training_size, &bit_rate);
        // Write results
        free(training_sequence);
        free(q_points);
        return EXIT_SUCCESS;
    } else {
        return EXIT_FAILURE;
    }
}