#include <random>
#include <unordered_set>
#include <float.h>
#include <iostream>
#include <cstring>

#define TRAINING_SIZE (1 << 20)
#define MAX_CODEBOOK_SIZE 256
#define BLOCK_SIZE2 64
#define POLYA_DELTA 0.00
#define POLYA_EPSILON 0.00
#define RATE 8
#define THRESHOLD 0.001

/**
 * Return an array of size TRAINING_SIZE containing values distributed according to N(0,1)
*/
double* generate_normal_sequence() {
    double* normal_sequence = (double*) malloc(TRAINING_SIZE * sizeof(double));
    std::default_random_engine rng;
    rng.seed(31);
    std::normal_distribution<double> distribution(0, 1);
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
 * @return double
 */
inline double error(double a, double b) {
  return (a - b) * (a - b);
}

inline double polya_urn_error(int j, int i, int num_bits) {
  double temp;
  int x = j ^ i;
  int previous;
  if(x & 1 == 1) {
    temp = POLYA_EPSILON;
    previous = 1;
  } else {
    temp = 1 - POLYA_EPSILON;
    previous = 0;
  }
  x >>= 1;
  for(int i = 1; i < num_bits; i++) {
    if(x & 1 == 1) {
      temp *= (POLYA_EPSILON + previous * POLYA_DELTA) / (1 + POLYA_DELTA);
      previous = 1;
    } else {
      temp *= ((1 - POLYA_EPSILON) + (1 - previous)*POLYA_DELTA) / (1 + POLYA_DELTA);
      previous = 0;
    }
    x >>= 1;
  }
  return temp;
}

double* compute_error_matrix(unsigned int levels) {
  double* error_matrix = (double*) malloc(sizeof(double) * levels * levels);
  for(int i = 0; i < levels; i++) {
      for(int j = 0; j < levels; j++) {
          error_matrix[j + i * levels] = polya_urn_error(j, i, RATE);
      }
  }
  return error_matrix;
}

void nnc(unsigned int* cells, double* training_sequence, double* codebook, int levels, double* error_matrix,
    double* cc_sums, unsigned int* cc_cardinality) {
  double min = __FLT_MAX__;
  int min_index = -1;
  double sum = 0;
  double c = 0;
  for(int i = 0; i < TRAINING_SIZE; i++) {
    double target = training_sequence[i];
    for(int l = 0; l < levels; l++) {
      // Kahan summation
      for(int j = 0; j < levels; j++) {
        double y = error_matrix[levels*l + j] * (target - codebook[j]) * (target - codebook[j]) - c;
        double t = sum + y;
        c = (t - sum) - y;
        sum = t;
      }
      if(sum < min) {
        min_index = l;
        min = sum;
      }
      sum=0;
      c=0;
    }
    cells[i] = min_index;
    // For Centroid Condition
    cc_cardinality[min_index]++; // update count
    cc_sums[min_index] += target; // running sum
    sum = 0;
    min_index = -1;
    min = __FLT_MAX__;
  }
}

void cc(int levels, double* error_matrix, double* cc_sums, unsigned int* cc_cardinality, double* codebook) {
  double numerator = 0;
  double denominator = 0;
  for (int j = 0; j < levels; j++) {
      for (int i = 0; i < levels; i++) {
          numerator += error_matrix[j + levels * i] * cc_sums[i];
      }
      for (int i = 0; i < levels; i++) {
          denominator += error_matrix[j + levels * i] * cc_cardinality[i];
      }
      codebook[j] = numerator / denominator;
      numerator = 0;
      denominator = 0;
  }
}

double distortion(unsigned int levels, double* training_sequence, double* error_matrix, double* codebook, unsigned int* cells) {
  double d = 0;
  double c = 0;
  for(int i = 0; i < TRAINING_SIZE; i++) {
    for(int j = 0; j < levels; j++) {
      double y = error_matrix[j + levels*cells[i]] * (training_sequence[i] - codebook[j]) * (training_sequence[i] - codebook[j]) - c;
      double t = d + y;
      c = (t - d) - y;
      d = t;
    }
  }
  return d / TRAINING_SIZE;
}


/**
 * @brief TODO: DOUBLE CHECK THIS
 *
 * @param training_sequence
 * @param training_size
 * @param rate
 * @param channel_error
 * @return double
 */
double singleton_cc(double* training_sequence, int training_size, int rate, double (*channel_error)(int, int, int)) {
  double sum = 0;
  for(int i = 0; i < training_size; i++)
    sum += training_sequence[i];
  return sum / training_size;
}

// /**
//  * Splitting technique:
//  * - A study of vector quantization for noisy channels, pg. 806 B.
//  * - An Algorithm for Vector Quantizer Design pg. 89
// */
// double* split(double* training_sequence, int training_size, int rate, double (*channel_error)(int, int, int)) {
//   double delta = 0.01;
//   double* temp = NULL;
//   double* codebook = (double*) malloc(sizeof(double) * (1 << rate));
//   double* temp1 = (double*) malloc(sizeof(double) * (1 << rate));
//   int k = 0;
//   int levels = 1;
//   cell* roots[MAX_CODEBOOK_SIZE];
//   cell* regions = (cell*) malloc(sizeof(cell) * TRAINING_SIZE);
//   for(int i = 0; i < TRAINING_SIZE; i++) {
//     regions[i].value = &training_sequence[i];
//     regions[i].next = NULL;
//   }
//   // Compute centroid of training sequence
//   codebook[0] = singleton_cc(training_sequence, training_size, rate, channel_error);
//   // Splitting loop
//   for(int i = 0; i < rate; i++) {
//     nearest_neighbour(codebook, roots, levels, regions, training_size, rate, channel_error);
//     centroid(roots, codebook, levels, rate, channel_error);
//     // Split!
//     for(int i = 0; i < levels; i++) {
//       temp1[2*i] = codebook[i] - delta;
//       temp1[2*i+1] = codebook[i] + delta;
//     }
//     // swap ptrs
//     temp = codebook;
//     codebook = temp1;
//     temp1 = temp;
//     levels <<= 1;
//   }
//   free(regions);
//   free(temp1);
//   std::cout << "Split generated codebook: [";
//   for(int i = 0; i < levels - 1; i++)
//     std::cout << codebook[i] << ", ";
//   std::cout << codebook[levels - 1] << "]" << std::endl;
//   return codebook;
// }

inline void swap(double* a, double* b) {
  double temp = *a;
  *a = *b;
  *b = temp;
}

/**
 * Steven S. Skiena
*/
inline void random_permutation(double* arr, int length) {
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
inline void perturb(unsigned int* x, unsigned int* y, double* arr, int levels) {
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
void printArrays(double* arr1, double* arr2, int levels) {
  std::cout << "Current: [";
  for(int i = 0; i < levels - 1; i++)
    std::cout << arr1[i] << ", ";
  std::cout << arr1[levels - 1] << "]" << std::endl;
  std::cout << "Perturbed: [";
  for(int i = 0; i < levels - 1; i++)
    std::cout << arr2[i] << ", ";
  std::cout << arr2[levels - 1] << "]" << std::endl;
}

// /**
//  * TODO: Still have NOT added condition where we exit if system "Appears stable" since I cannot
//  * find any good resource on how this is determined in any of the research papers.
//  *
//  * Modelled primarily using "A_study_of_vector_quantization_for_noisy_channels"
//  * To get algorithm skeleton, "Using Simulated Annealing to Design Good Codes"
// */
// double* simulated_annealing(double* training_sequence, int training_size, int rate, double (*channel_error)(int, int, int)) {
//   double* current = split(training_sequence, training_size, rate, channel_error);
//   unsigned int levels = 1 << rate;
//   double* next = (double*) malloc(sizeof(double) * levels);
//   cell* roots[MAX_CODEBOOK_SIZE];  // roots is used to point to the beginning of the linked list.
//   cell* regions = (cell*) malloc(sizeof(cell) * TRAINING_SIZE);
//   double temperature = 10;
//   // const double final_temp = 0.00025;
//   // const double alpha = 0.97;
//   const double final_temp = 8;
//   const double alpha = 0.97;
//   double d1 = 0, d2 = 0, delta = 0;
//   // end conditions
//   const unsigned int max_iterations = 10;
//   unsigned int i = 0;
//   const unsigned int max_drops = 5;
//   unsigned int drops = 0;
//   // swapping
//   unsigned int x, y;
//   std::default_random_engine generator;
//   std::uniform_real_distribution<double> distr(0.0, 1.0);
//   //setup regions
//   for(int i = 0; i < training_size; i++) {
//     regions[i].value = &training_sequence[i];
//     regions[i].next = NULL;
//   }
//   std::memcpy(next, current, sizeof(double) * levels);
//   do {
//     do {
//       std::cout << "Iteration: " << i << std::endl;
//       // Generate perturbed index assignment
//       perturb(&x, &y, next, levels);
//       std::cout << "After Perturbation:" << std::endl;
//       printArrays(current, next, levels);
//       // Calculate distortion
//       nearest_neighbour(current, roots, levels, regions, training_size, rate, channel_error);
//       d1 = distortion(levels, rate, roots, current, training_size, channel_error);
//       nearest_neighbour(next, roots, levels, regions, training_size, rate, channel_error);
//       d2 = distortion(levels, rate, roots, next, training_size, channel_error);
//       delta = d2 - d1;
//       if(delta < 0) {
//         drops++;
//         // Make current the same as next.
//         if(y >= x) {
//           swap(current + x, current + y + 1);
//         } else {
//           swap(current + x, current + y);
//         }
//       } else if(distr(generator) <= exp(-1*delta/temperature)) {
//         // Make current the same as next.
//         if(y >= x) {
//           swap(current + x, current + y + 1);
//         } else {
//           swap(current + x, current + y);
//         }
//       }
//       std::cout << "After energy measurement:" << std::endl;
//       printArrays(current, next, levels);
//       i++;
//     } while (max_drops > drops && i < max_iterations);
//     // decrease temp
//     temperature *= alpha;
//     std::cout << "::::::::::::::::::::::::TEMPERATURE IS NOW:" << temperature << " :::::::::::::::::::::::::::" << std::endl;
//     drops = 0;
//     i = 0;
//   } while(temperature > final_temp);
//   free(next);
//   free(regions);
//   return current;
// }

void cosq(double* training_sequence) {
  int levels = 1 << RATE;
  double current_distortion = 0, previous_distortion = DBL_MAX;
  double* error_matrix = compute_error_matrix(levels);
  double* codebook = (double*) malloc(sizeof(double) * levels);
  unsigned int* cells = (unsigned int*) malloc(sizeof(unsigned int) * TRAINING_SIZE);
  double* cc_cell_sums = (double*) malloc(sizeof(double) * levels);
  unsigned int* cc_cell_cardinality = (unsigned int*) malloc(sizeof(unsigned int) * levels);
  for(int i = 0; i < levels; i++)
    codebook[i] = training_sequence[i];
  // codebook = simulated_annealing(training_sequence, TRAINING_SIZE, RATE, channel_error)s;
  // Lloyd Iteration
  while(fabsf64(previous_distortion - current_distortion) / previous_distortion > THRESHOLD) {
    nnc(cells, training_sequence, codebook, levels, error_matrix, cc_cell_sums, cc_cell_cardinality);
    cc(levels, error_matrix, cc_cell_sums, cc_cell_cardinality, codebook);
    current_distortion = distortion(levels, training_sequence, error_matrix, codebook, cells);
    std::cout << "Current distortion is " << current_distortion << std::endl;
    previous_distortion = current_distortion;
  }
  std::cout << "Results! [";
  for(int i = 0; i < levels - 1; i++)
    std::cout << codebook[i] << ", ";
  std::cout << codebook[levels - 1] << "]" << std::endl;
  std::cout << "Distortion is: " << current_distortion << std::endl;
  free(codebook);
  free(error_matrix);
}

int main(int argc, char** argv) {
  printf("Training %d-bit normal quantizer for polya channel delta %f, epsilon %f\n", RATE, POLYA_DELTA, POLYA_EPSILON);
  double* normal = generate_normal_sequence();
  // Verify 0 mean and unit variance::
  double sum = 0;
  double n_avg = 0;
  double n_var = 0;
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
  cosq(normal);
  free(normal);
  return 0;
}