#include <random>
#include <stdlib.h>
#include <iostream>
/**
 * Return an array of size TRAINING_SIZE containing values distributed according to N(0,1)
*/
double* generate_normal_sequence(unsigned int* training_size) {
    double* normal_sequence = (double*) malloc(*training_size * sizeof(double));
    if(normal_sequence == nullptr) {
        std::cout << "Memory Allocation error: Failed to allocate memory for normal_sequence!" << std::endl;
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

int main(int argc, char** argv) {
    unsigned int training_size = atoi(argv[1]);
    double* seq = generate_normal_sequence(&training_size);
    FILE* training_seq_file = fopen("sequence", "w");
    size_t count = fwrite(seq, sizeof(double), training_size, training_seq_file);
    if(count != training_size) {
        std::cout << "Error: cannot write entire training sequence to file!" << std::endl;
    }
    fclose(training_seq_file);
}