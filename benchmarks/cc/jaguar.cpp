#include <random>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <omp.h>

#define TRAINING_SIZE 1048576
#define RATE 8
#define POLYA_EPSILON 0.01
#define POLYA_DELTA 0
#define FLOAT_ERROR 1

void cc_cpu(int levels, float* error_matrix, float* cc_sums, unsigned int* cc_cardinality, float* codebook) {
  float numerator = 0;
  float denominator = 0;;
  for(int j = 0; j < levels; j++) {
    for(int i = 0; i < levels; i++) {
      numerator += error_matrix[j + levels*i] * cc_sums[i];
    }
    for(int i = 0; i < levels; i++) {
      denominator += error_matrix[j + levels*i] * cc_cardinality[i];
    }
    codebook[j] = numerator / denominator;
    numerator = 0;
    denominator = 0;
  }
}

void omp_cc_cpu(int levels, float* error_matrix, float* cc_sums, unsigned int* cc_cardinality, float* codebook) {
  omp_set_num_threads(12);
  #pragma omp parallel
  {
    unsigned int id = omp_get_thread_num();
    unsigned int num_threads = omp_get_num_threads();
    unsigned int sums_per_thread = levels / num_threads;
    unsigned int remainder = levels - sums_per_thread * num_threads;
    int lvl = remainder - id;
    unsigned int start = lvl > 0 ? id * (sums_per_thread + 1) : remainder * (sums_per_thread + 1) + (id - remainder) * sums_per_thread;
    unsigned int end = lvl > 0 ? start + sums_per_thread : start + sums_per_thread - 1;
    float numerator = 0;
    float denominator = 0;
    for(int j = start; j <= end; j++) {
      for(int i = 0; i < levels; i++) {
        numerator += error_matrix[j + levels*i] * cc_sums[i];
        denominator += error_matrix[j + levels*i] * cc_cardinality[i];
      }
      codebook[j] = numerator / denominator;
      numerator = 0;
      denominator = 0;
    }
  }
}

inline float polya_urn_error(int j, int i, int num_bits) {
  float temp;
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

float* compute_error_matrix(unsigned int levels) {
  float* error_matrix = (float*) malloc(sizeof(float) * levels * levels);
  for(int i = 0; i < levels; i++) {
      for(int j = 0; j < levels; j++) {
          error_matrix[j + i * levels] = polya_urn_error(j, i, RATE);
      }
  }
  return error_matrix;
}

/**
 * Return an array of size TRAINING_SIZE containing values distributed according to N(0,1)
*/
float* generate_normal_sequence() {
  float* normal_sequence = (float*) malloc(TRAINING_SIZE * sizeof(float));
  std::default_random_engine rng;
  rng.seed(31);
  std::normal_distribution<float> distribution(10, 1);
  for(int i = 0; i < TRAINING_SIZE; i++) {
      normal_sequence[i] = distribution(rng);
  }
  return normal_sequence;
}

void cc_correct(float* codebook_seq, float* codebook_mp, unsigned int levels) {
  bool correct = true;
  for(int i = 0; i < levels; i++) {
    if(codebook_seq[i] != codebook_mp[i]) {
      printf("The codebooks DO NOT match!\n");
      printf("Disagreement at %d: codebook_seq %f, codebook mp %f", i, codebook_seq[i], codebook_mp[i]);
      correct = false;
      break;
    }
  }
  if(correct)
    printf("The codebooks match! CC Correctness test passed!\n");
}

int main(int argc, char** argv) {
  const unsigned int levels = 1 << RATE;
  float* training_sequence = generate_normal_sequence();
  float* error_matrix = compute_error_matrix(levels);
  float* codebook_seq = (float*) malloc(sizeof(float) * levels);
  float* codebook_mp = (float*) malloc(sizeof(float) * levels);
  float* cc_training_sums = (float*) calloc(levels, sizeof(float));
  unsigned int* cc_cardinality = (unsigned int*) calloc(levels, sizeof(unsigned int));
  // intialize codebook to first <levels> training samples
  // initialize training_sums and cc_cardinality
  std::default_random_engine rng;
  std::uniform_int_distribution<int> distribution(1, 100);
  rng.seed(31);
  for(int i = 0; i < levels; i++) {
    cc_training_sums[i] = (float) distribution(rng);
    cc_cardinality[i] = (float) distribution(rng);
  }
  /*****************************************************************************************
   * Tests for Centroid Condition
  *****************************************************************************************/

  // Accuracy test
  printf(":::::::::::: Performing correctness test CC ::::::::::::\n");
  cc_cpu(levels, error_matrix, cc_training_sums, cc_cardinality, codebook_seq);
  omp_cc_cpu(levels, error_matrix, cc_training_sums, cc_cardinality, codebook_mp);
  cc_correct(codebook_seq, codebook_mp, levels);

  /*
    Sequential CC
  */
  unsigned long int sum = 0;
  for(int i = 0; i < 100; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    cc_cpu(levels, error_matrix, cc_training_sums, cc_cardinality, codebook_seq);
    auto end = std::chrono::high_resolution_clock::now();
    auto t = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    sum += t.count();
  }
  std::cout << ":::::::::::: Performance CPU-only code ::::::::::::" << std::endl;
  std::cout << "sequential result took " << sum / 100 << "ns on average (100 iters)." << std::endl;

  /*
    OpenMP CC
  */

  sum = 0;
  int avg = 0;
  for(int i = 0; i < 100; i++) {
    auto start = std::chrono::high_resolution_clock::now();
    omp_cc_cpu(levels, error_matrix, cc_training_sums, cc_cardinality, codebook_mp);
    auto end = std::chrono::high_resolution_clock::now();
    auto t = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    sum += t.count();
    if(i != 0) // ignore warmup
      avg += t.count();
  }
  std::cout << ":::::::::::: Performance OMP CPU-only code ::::::::::::" << std::endl;
  std::cout << "OMP result took " << sum / 100 << "ns on average (100 iters). " << "After warmup (1), avg is " << avg /99 << "ns." << std::endl;
  free(codebook_seq);
  free(codebook_mp);
  free(cc_cardinality);
  free(cc_training_sums);
  free(training_sequence);
  free(error_matrix);
}