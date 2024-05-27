#include <random>
#include <unordered_set>
#include <iostream>

#define TRAINING_SIZE 100000
#define MAX_CODEBOOK_SIZE 256
#define BLOCK_SIZE2 64

// Binary Symmetric Channel

// Global variables: BSC
float bsc_crossover = 0;

struct quantizer_cell {
  struct quantizer_cell* next;
  float* value;
};

typedef struct quantizer_cell cell;

void nullify(cell** arr, int length) {
  for(int i = 0; i < length; i++)
    arr[i] = NULL;
}

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

/**
 * @brief Error measure used by the cosq.
 *
 * @param a
 * @param b
 * @return float
 */
float error(float a, float b) {
  return (a - b) * (a - b);
}

void compute_error_matrix(float* error_matrix, unsigned int length, unsigned int num_bits, float (*channel_error)(int, int, int)) {
  for(int i = 0; i < length; i++) {
    for(int j = 0; j < length; j++) {
      error_matrix[j + i * length] = channel_error(i, j, num_bits);
    }
  }
}

/**
 * @brief Channel error probability for the Binary Symmetric Channel.
 *
 */
float bsc_error(int a, int b, int num_bits) {
  int x = a ^ b;
  int count = 0; // number of bits that differ
  while (x) {
    count += x & 1;
    x >>= 1;
  }
  return pow(bsc_crossover, count) * pow(1-bsc_crossover, num_bits - count);
}

// float polya_urn_error(int a, int b, int num_bits) {
//   float temp;
//   int x = a ^ b;
//   int previous;
//   if(x & 1 == 1) {
//     temp = polya_epsilon;
//     previous = 1;
//   } else {
//     temp = 1 - polya_epsilon;
//     previous = 0;
//   }
//   x >>= 1;
//   for(int i = 1; i < num_bits; i++) {
//     if(x & 1 == 1) {
//       temp *= (polya_epsilon + previous * polya_delta) / (1 + polya_delta);
//       previous = 1;
//     } else {
//       temp *= ((1 - polya_epsilon) + (1 - previous)*polya_delta) / (1 + polya_delta);
//       previous = 0;
//     }
//     x >>= 1;
//   }
//   return temp;
// }

/**
 * @brief Generalized nearest neighbour condition.
 *
 * @param codebook
 * @param regions
 * @return float
 */
void nearest_neighbour(float* codebook, cell** roots, int levels, cell* regions, int training_size, int num_bits, float* error_matrix) {
  float min = __FLT_MAX__;
  int min_index = -1;
  float sum = 0;
  cell* previous[MAX_CODEBOOK_SIZE];
  nullify(previous, levels);
  for(int i = 0; i < training_size; i++) {
    // printf("Iteration %d\n", i);
    for(int l = 0; l < levels; l++) {
      for(int j = 0; j < levels; j++) {
        // if(l == j)
        //     sum += error(*(regions[i].value), codebook[j]);
        sum += error_matrix[j + l * levels] * error(*(regions[i].value), codebook[j]);
      }
      if(sum < min) {
        min_index = l;
        min = sum;
      }
      sum=0;
    }
    // If first node in this partition:
    if(previous[min_index] == NULL) {
      roots[min_index] = &regions[i];
      previous[min_index] = roots[min_index];
    } else {
      (*previous[min_index]).next = &regions[i];
      previous[min_index] = &regions[i];
    }
    regions[i].next = NULL;
    //reset
    sum = 0;
    min_index = -1;
    min = __FLT_MAX__;
  }
}

/**
 * @brief Generalized centroid condition.
 *
 * @param regions
 * @param codebook
 * @return float
 */
void centroid(cell** roots, float* codebook, int levels, int num_bits, float* error_matrix) {
  float numerator = 0;
  float denominator = 0;
  float partition_sum = 0;
  float partition_sizes[MAX_CODEBOOK_SIZE];
  float partition_sums[MAX_CODEBOOK_SIZE];
  cell* ptr = NULL;
  int count = 0;

  // To save ourselves from calculating the same sum
  // for each codebook value, calculate it once and save the value.
  for(int i = 0; i < levels; i++) {
    ptr = roots[i];
    while(ptr != NULL) {
      partition_sum += *((*ptr).value);
      count++;
      ptr = ptr->next;
    }
    partition_sizes[i] = count;
    partition_sums[i] = partition_sum;
    count = 0;
    partition_sum = 0;
  }

  for(int i = 0; i < levels; i++) {
    // Compute Numerator
    for(int j = 0; j < levels; j++)
      numerator += error_matrix[j + i * levels] * partition_sums[j];
    // Compute Denominator
    for(int j = 0; j < levels; j++)
      denominator += error_matrix[j + i * levels] * partition_sizes[j];
    if(denominator == 0)
      std::cout << "\nERROR: DIVIDE BY ZERO!!!!\n";
    codebook[i] = numerator/denominator;
    numerator = 0;
    denominator = 0;
  }
}

/**
 * @brief
 *
 * @param levels
 * @param num_bits
 * @param training
 * @param codebook
 * @return float
 */
float distortion(int levels, int num_bits, cell** roots, float* codebook, int training_size, float* error_matrix) {
  float d = 0;
  cell* traversal = NULL;
  for(int i = 0; i < levels; i++) {
    traversal = roots[i];
    while(traversal != NULL) {
      for(int j = 0; j < levels; j++) {
        d += error_matrix[j + i * levels] * error(*(traversal->value), codebook[j]);
      }
      traversal = traversal->next;
    }
  }
  return d / training_size;
}

void cosq(float* training_sequence, int num_bits, float* quantizers, float (*channel_error)(int, int, int)) {
  std::cout << "Training quantizer with bitrate " << num_bits << std::endl;
  int levels = 1 << num_bits;
  float current_distortion = 0;
  float previous_distortion = 0;
  float* codebook = quantizers;
  cell* roots[MAX_CODEBOOK_SIZE];  // roots is used to point to the beginning of the linked list.
  cell* regions = (cell*) malloc(sizeof(cell) * TRAINING_SIZE);
  float error_matrix[levels * levels];
  compute_error_matrix(error_matrix, levels, num_bits, channel_error);
  nullify(roots, MAX_CODEBOOK_SIZE);
  float threshold = 0.0001;
  // normal quantizer
  //setup regions
  for(int i = 0; i < TRAINING_SIZE; i++) {
      regions[i].value = &training_sequence[i];
      regions[i].next = NULL;
  }
  // initialize codebook
  // Use first N training points as initial codebook.
  for(int i = 0; i < levels; i++) {
      codebook[i] = training_sequence[i];
  }
  // First iteration
  nearest_neighbour(codebook, roots, levels, regions, TRAINING_SIZE, num_bits, error_matrix);
  centroid(roots, codebook, levels, num_bits, error_matrix);
  previous_distortion = distortion(levels, num_bits, roots, codebook, TRAINING_SIZE, error_matrix);
  // Lloyd Iteration
  while(1) {
    nearest_neighbour(codebook, roots, levels, regions, TRAINING_SIZE, num_bits, error_matrix);
    centroid(roots, codebook, levels, num_bits, error_matrix);
    current_distortion = distortion(levels, num_bits, roots, codebook, TRAINING_SIZE, error_matrix);
    if((previous_distortion - current_distortion) / previous_distortion < threshold)
      break;
    previous_distortion = current_distortion;
  }
  std::cout << "Results! [";
  for(int i = 0; i < levels - 1; i++)
    std::cout << codebook[i] << ", ";
  std::cout << codebook[levels - 1] << "]" << std::endl;
  std::cout << "Distortion is: " << current_distortion << std::endl;
}

int main(int argc, char** argv) {
    bsc_crossover = atof(argv[1]);
    int normal_bitrate = atoi(argv[2]);
    int laplacian_bitrate = atoi(argv[3]);
    printf("Training %d bit laplacian, %d bit normal quantizers for bsc channel error %f\n", laplacian_bitrate, normal_bitrate, bsc_crossover);

    // Delta 5, 10
    // Epsilon 0.005, 0.01 0.1
    float* quantizers = NULL;
    // First entry is normally distributed
    float* normal = generate_normal_sequence();
    float* laplace = generate_laplacian_sequence();
    // Verify 0 mean and unit variance::
    float sum = 0;
    float n_avg = 0;
    float n_var = 0;
    for(int i = 0; i < TRAINING_SIZE; i++)
        sum += normal[i];
    n_avg = sum / TRAINING_SIZE;

    std::cout << "Normal sequence E[X] = " << n_avg << std::endl;

    // Variance
    sum = 0;
    for(int i = 0; i < TRAINING_SIZE; i++)
        sum += pow(normal[i] - n_avg, 2);
    n_var = sum/TRAINING_SIZE;
    std::cout << "Normal sequence Var(X) = " << n_var << std::endl;

    sum = 0;
    float l_avg = 0;
    float l_var = 0;
    for(int i = 0; i < TRAINING_SIZE; i++)
        sum += laplace[i];
    l_avg = sum / TRAINING_SIZE;

    std::cout << "Laplacian sequence E[X] = " << l_avg << std::endl;
    // Variance
    sum = 0;
    for(int i = 0; i < TRAINING_SIZE; i++)
        sum += pow(laplace[i] - l_avg, 2);
    l_var = sum/TRAINING_SIZE;

    std::cout << "Laplacian sequence Var(X) = " << l_var << std::endl;

    // Start training here
    quantizers = (float*) malloc(MAX_CODEBOOK_SIZE * 2 * sizeof(float));
    cosq(normal, normal_bitrate, quantizers, bsc_error);
    cosq(laplace, laplacian_bitrate, quantizers +  MAX_CODEBOOK_SIZE, bsc_error);
    free(normal);
    free(laplace);
    return 0;
}