#include <random>
#include <iostream>

#define TRAINING_SIZE 1000000
#define MAX_CODEBOOK_SIZE 256

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

/**
 * @brief Channel error probability for the Binary Symmetric Channel.
 *
 */
float channel_error(int a, int b, int num_bits) {
  int x = a ^ b;
  int count = 0; // number of bits that differ
  float p = 0; // probability of error
  while (x) {
    count += x & 1;
    x >>= 1;
  }
  return pow(p, count) * pow(1-p, num_bits - count);
}

/**
 * @brief Generalized nearest neighbour condition.
 *
 * @param codebook
 * @param regions
 * @return float
 */
void nearest_neighbour(float* codebook, cell** roots, int levels, cell* regions, int training_size, int num_bits) {
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
        sum += channel_error(j, l, num_bits) * error(*(regions[i].value), codebook[j]);
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
void centroid(cell** roots, float* codebook, int levels, int num_bits) {
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
      numerator += channel_error(i, j, num_bits) * partition_sums[j];
    // Compute Denominator
    for(int j = 0; j < levels; j++)
      denominator += channel_error(i, j, num_bits) * partition_sizes[j];
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
float distortion(int levels, int num_bits, cell** roots, float* codebook, int training_size) {
  float d = 0;
  cell* traversal = NULL;
  for(int i = 0; i < levels; i++) {
    traversal = roots[i];
    while(traversal != NULL) {
      for(int j = 0; j < levels; j++) {
        d += channel_error(j, i, num_bits) * error(*(traversal->value), codebook[j]);
      }
      traversal = traversal->next;
    }
  }
  return d / training_size;
}

void cosq(
        float* normal_sequence,
        int num_bits) {
    int levels = 1 << num_bits;
    float current_distortion = 0;
    float previous_distortion = 0;
    float codebook[MAX_CODEBOOK_SIZE]; // we will only use indexes up to int levels.
    cell* roots[MAX_CODEBOOK_SIZE];  // roots is used to point to the beginning of the linked list.
    cell* regions = (cell*) malloc(sizeof(cell) * TRAINING_SIZE);
    nullify(roots, MAX_CODEBOOK_SIZE);
    float threshold = 0.01;
    // normal quantizer
    //setup regions
    for(int i = 0; i < TRAINING_SIZE; i++) {
        regions[i].value = &normal_sequence[i];
        regions[i].next = NULL;
    }
    // initialize codebook
    // Use first N training points as initial codebook.
    for(int i = 0; i < levels; i++) {
        codebook[i] = normal_sequence[i];
    }
    // First iteration
    nearest_neighbour(codebook, roots, levels, regions, TRAINING_SIZE, num_bits);
    centroid(roots, codebook, levels, num_bits);
    previous_distortion = distortion(levels, num_bits, roots, codebook, TRAINING_SIZE);
    // Lloyd Iteration
    while(1) {
      nearest_neighbour(codebook, roots, levels, regions, TRAINING_SIZE, num_bits);
      centroid(roots, codebook, levels, num_bits);
      current_distortion = distortion(levels, num_bits, roots, codebook, TRAINING_SIZE);
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

int main() {
    float* normal = generate_normal_sequence();
    cosq(normal, 8);
    free(normal);
    return 0;
}