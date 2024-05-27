#include <random>
#include <unordered_set>
#include <iostream>
#include <cstring>

#define TRAINING_SIZE 100000
#define MAX_CODEBOOK_SIZE 256
#define BLOCK_SIZE2 64

// Global variables: Polya urn Channel
float polya_epsilon = 0;
float polya_delta = 0;

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
 * @brief Error measure used by the cosq.
 *
 * @param a
 * @param b
 * @return float
 */
inline float error(float a, float b) {
  return (a - b) * (a - b);
}

// void compute_error_matrix(float* error_matrix, unsigned int length, unsigned int num_bits, float (*channel_error)(int, int, int)) {
//   for(int i = 0; i < length; i++) {
//     for(int j = 0; j < length; j++) {
//       error_matrix[j + i * length] = channel_error(i, j, num_bits);
//     }
//   }
// }

float polya_urn_error(int a, int b, int num_bits) {
  float temp;
  int x = a ^ b;
  int previous;
  if(x & 1 == 1) {
    temp = polya_epsilon;
    previous = 1;
  } else {
    temp = 1 - polya_epsilon;
    previous = 0;
  }
  x >>= 1;
  for(int i = 1; i < num_bits; i++) {
    if(x & 1 == 1) {
      temp *= (polya_epsilon + previous * polya_delta) / (1 + polya_delta);
      previous = 1;
    } else {
      temp *= ((1 - polya_epsilon) + (1 - previous)*polya_delta) / (1 + polya_delta);
      previous = 0;
    }
    x >>= 1;
  }
  return temp;
}

/**
 * @brief Generalized nearest neighbour condition.
 *
 * @param codebook
 * @param regions
 * @return float
 */
void nearest_neighbour(float* codebook, cell** roots, int levels, cell* regions, int training_size, int num_bits, float (*channel_error)(int, int, int)) {
  float min = __FLT_MAX__;
  int min_index = -1;
  float sum = 0;
  cell* previous[MAX_CODEBOOK_SIZE];
  nullify(previous, levels);
  for(int i = 0; i < training_size; i++) {
    for(int l = 0; l < levels; l++) {
      for(int j = 0; j < levels; j++) {
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
void centroid(cell** roots, float* codebook, int levels, int num_bits, float (*channel_error)(int, int, int)) {
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
      numerator += channel_error(j, i, num_bits) * partition_sums[j];
    // Compute Denominator
    for(int j = 0; j < levels; j++)
      denominator += channel_error(j, i, num_bits) * partition_sizes[j];
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
float distortion(int levels, int num_bits, cell** roots, float* codebook, int training_size, float (*channel_error)(int, int, int)) {
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

/**
 * @brief TODO: DOUBLE CHECK THIS
 *
 * @param training_sequence
 * @param training_size
 * @param rate
 * @param channel_error
 * @return float
 */
float singleton_cc(float* training_sequence, int training_size, int rate, float (*channel_error)(int, int, int)) {
  float sum = 0;
  for(int i = 0; i < training_size; i++)
    sum += training_sequence[i];
  return sum / training_size;
}

/**
 * Splitting technique:
 * - A study of vector quantization for noisy channels, pg. 806 B.
 * - An Algorithm for Vector Quantizer Design pg. 89
*/
float* split(float* training_sequence, int training_size, int rate, float (*channel_error)(int, int, int)) {
  float delta = 0.01;
  float* temp = NULL;
  float* codebook = (float*) malloc(sizeof(float) * (1 << rate));
  float* temp1 = (float*) malloc(sizeof(float) * (1 << rate));
  int k = 0;
  int levels = 1;
  cell* roots[MAX_CODEBOOK_SIZE];
  cell* regions = (cell*) malloc(sizeof(cell) * TRAINING_SIZE);
  for(int i = 0; i < TRAINING_SIZE; i++) {
    regions[i].value = &training_sequence[i];
    regions[i].next = NULL;
  }
  // Compute centroid of training sequence
  codebook[0] = singleton_cc(training_sequence, training_size, rate, channel_error);
  // Splitting loop
  for(int i = 0; i < rate; i++) {
    nearest_neighbour(codebook, roots, levels, regions, training_size, rate, channel_error);
    centroid(roots, codebook, levels, rate, channel_error);
    // Split!
    for(int i = 0; i < levels; i++) {
      temp1[2*i] = codebook[i] - delta;
      temp1[2*i+1] = codebook[i] + delta;
    }
    // swap ptrs
    temp = codebook;
    codebook = temp1;
    temp1 = temp;
    levels <<= 1;
  }
  free(regions);
  free(temp1);
  std::cout << "Split generated codebook: [";
  for(int i = 0; i < levels - 1; i++)
    std::cout << codebook[i] << ", ";
  std::cout << codebook[levels - 1] << "]" << std::endl;
  return codebook;
}

inline void swap(float* a, float* b) {
  float temp = *a;
  *a = *b;
  *b = temp;
}

/**
 * Steven S. Skiena
*/
inline void random_permutation(float* arr, int length) {
  std::random_device rd;
  std::mt19937 gen(rd());
  for(int i = 0; i < length; i++) {
    std::uniform_int_distribution<> distr(i, length - 1);
    swap(arr + i, arr + distr(gen));
  }
}

/**
 * Described in "A_study_of_vector_quantization_for_noisy_channels"
*/
inline void perturb(unsigned int* x, unsigned int* y, float* arr, int levels) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> first(0, levels - 1);
  std::uniform_int_distribution<> second(0, levels - 2);
  *x = first(gen);
  *y = second(gen);
  if(*y >= *x) {
    swap(arr + *x, arr + *y + 1);
  } else {
    swap(arr + *x, arr + *y);
  }
  // std::cout << "Swapping " << *x << " and " << *y << std::endl;
  // std::cout << "Perturbed inside func: [";
  // for(int i = 0; i < levels - 1; i++)
  //   std::cout << arr[i] << ", ";
  // std::cout << arr[levels - 1] << "]" << std::endl;
}

/**
 * 
 * TODO: DELETE AFTER DEBUGGING!!
*/
void printArrays(float* arr1, float* arr2, int levels) {
  std::cout << "Current: [";
  for(int i = 0; i < levels - 1; i++)
    std::cout << arr1[i] << ", ";
  std::cout << arr1[levels - 1] << "]" << std::endl;
  std::cout << "Perturbed: [";
  for(int i = 0; i < levels - 1; i++)
    std::cout << arr2[i] << ", ";
  std::cout << arr2[levels - 1] << "]" << std::endl;
}

/**
 * TODO: Still have NOT added condition where we exit if system "Appears stable" since I cannot
 * find any good resource on how this is determined in any of the research papers.
 *
 * Modelled primarily using "A_study_of_vector_quantization_for_noisy_channels"
 * To get algorithm skeleton, "Using Simulated Annealing to Design Good Codes"
*/
float* simulated_annealing(float* training_sequence, int training_size, int rate, float (*channel_error)(int, int, int)) {
  float* current = split(training_sequence, training_size, rate, channel_error);
  unsigned int levels = 1 << rate;
  float* next = (float*) malloc(sizeof(float) * levels);
  cell* roots[MAX_CODEBOOK_SIZE];  // roots is used to point to the beginning of the linked list.
  cell* regions = (cell*) malloc(sizeof(cell) * TRAINING_SIZE);
  float temperature = 10;
  const float final_temp = 0.00025;
  const float alpha = 0.97;
  float d1 = 0, d2 = 0, delta = 0;
  // end conditions
  const unsigned int max_iterations = 10;
  unsigned int i = 0;
  const unsigned int max_drops = 5;
  unsigned int drops = 0;
  // swapping
  unsigned int x, y;
  std::default_random_engine generator;
  std::uniform_real_distribution<float> distr(0.0, 1.0);
  //setup regions
  for(int i = 0; i < training_size; i++) {
    regions[i].value = &training_sequence[i];
    regions[i].next = NULL;
  }
  std::memcpy(next, current, sizeof(float) * levels);
  do {
    do {
      std::cout << "Iteration: " << i << std::endl;
      // Generate perturbed index assignment
      perturb(&x, &y, next, levels);
      std::cout << "After Perturbation:" << std::endl;
      printArrays(current, next, levels);
      // Calculate distortion
      nearest_neighbour(current, roots, levels, regions, training_size, rate, channel_error);
      d1 = distortion(levels, rate, roots, current, training_size, channel_error);
      nearest_neighbour(next, roots, levels, regions, training_size, rate, channel_error);
      d2 = distortion(levels, rate, roots, next, training_size, channel_error);
      delta = d2 - d1;
      if(delta < 0) {
        drops++;
        // Make current the same as next.
        if(y >= x) {
          swap(current + x, current + y + 1);
        } else {
          swap(current + x, current + y);
        }
      } else if(distr(generator) <= exp(-1*delta/temperature)) {
        // Make current the same as next.
        if(y >= x) {
          swap(current + x, current + y + 1);
        } else {
          swap(current + x, current + y);
        }
      }
      std::cout << "After energy measurement:" << std::endl;
      printArrays(current, next, levels);
      i++;
    } while (max_drops > drops && i < max_iterations);
    // decrease temp
    temperature *= alpha;
    drops = 0;
    i = 0;
  } while(temperature > final_temp);
  free(next);
  return current;
}

void cosq(float* training_sequence, int rate, float (*channel_error)(int, int, int)) {
  std::cout << "Training quantizer with rate " << rate << std::endl;
  int levels = 1 << rate;
  float current_distortion = 0;
  float previous_distortion = 0;
  float* codebook;
  cell* roots[MAX_CODEBOOK_SIZE];  // roots is used to point to the beginning of the linked list.
  cell* regions = (cell*) malloc(sizeof(cell) * TRAINING_SIZE);
  float error_matrix[levels * levels];
  // compute_error_matrix(error_matrix, levels, rate, channel_error);
  nullify(roots, MAX_CODEBOOK_SIZE);
  float threshold = 0.0001;
  // normal quantizer
  //setup regions
  for(int i = 0; i < TRAINING_SIZE; i++) {
    regions[i].value = &training_sequence[i];
    regions[i].next = NULL;
  }
  codebook = simulated_annealing(training_sequence, TRAINING_SIZE, rate, channel_error);
  // First iteration
  // nearest_neighbour(codebook, roots, levels, regions, TRAINING_SIZE, rate, channel_error);
  // centroid(roots, codebook, levels, rate, channel_error);
  // previous_distortion = distortion(levels, rate, roots, codebook, TRAINING_SIZE, channel_error);
  // // Lloyd Iteration
  // while(1) {
  //   nearest_neighbour(codebook, roots, levels, regions, TRAINING_SIZE, rate, channel_error);
  //   centroid(roots, codebook, levels, rate, channel_error);
  //   current_distortion = distortion(levels, rate, roots, codebook, TRAINING_SIZE, channel_error);
  //   if((previous_distortion - current_distortion) / previous_distortion < threshold)
  //     break;
  //   previous_distortion = current_distortion;
  // }
  std::cout << "Results! [";
  for(int i = 0; i < levels - 1; i++)
    std::cout << codebook[i] << ", ";
  std::cout << codebook[levels - 1] << "]" << std::endl;
  std::cout << "Distortion is: " << current_distortion << std::endl;
  free(codebook);
  free(regions);
}

int main(int argc, char** argv) {
  polya_delta = atof(argv[1]);
  polya_epsilon = atof(argv[2]);
  int rate = atoi(argv[3]);
  printf("Training %d-bit normal quantizer for polya channel delta %f, epsilon %f\n", rate, polya_delta, polya_epsilon);
  float* normal = generate_normal_sequence();
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
  // Start training here
  cosq(normal, rate, polya_urn_error);
  free(normal);
  return 0;
}