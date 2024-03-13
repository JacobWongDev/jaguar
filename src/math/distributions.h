#include <random>
#include "../cuda/common.h"
/**
 * Return an array of size TRAINING_SIZE containing values distributed according to N(0,1)
*/
float* generate_normal_sequence() {
    float* normal_sequence = (float*) malloc(TRAINING_SIZE * sizeof(float));
    std::default_random_engine rng;
    rng.seed(31);
    std::normal_distribution<float> distribution(0, 1);
    for(int i = 0; i < TRAINING_SIZE; i++) {
        normal_sequence[i] = distribution(rng);
    }
    return normal_sequence;
}

/**
 * Return an array of size TRAINING_SIZE containing values distributed according to L(0, 1/4)
 *
 * https://www.math.wm.edu/~leemis/chart/UDR/PDFs/ExponentialLaplace.pdf
 * https://digitalcommons.usf.edu/cgi/viewcontent.cgi?article=3442&context=etd
 *
 * Can generate such a pdf using two independent exponentials A, B with parameters a = 1/sqrt(2), b = sqrt(2) - 1/sqrt(2)
 * where Y (laplacian pdf) = A - B
*/
float* generate_laplacian_sequence() {
    float* laplacian_sequence = (float*) malloc(TRAINING_SIZE * sizeof(float));
    std::exponential_distribution<float> x(sqrt(2));
    std::exponential_distribution<float> y(sqrt(2));
    std::random_device rd;
    std::mt19937 gen(rd());
    gen.seed(31);
    for(int i = 0; i < TRAINING_SIZE; i++) {
        laplacian_sequence[i] = x(gen) - y(gen);
    }
    return laplacian_sequence;
}