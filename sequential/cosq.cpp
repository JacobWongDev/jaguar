#include <random>
#include <unordered_set>
#include <float.h>
#include <iostream>
#include <cstring>
#include <chrono>

int training_size;
int rate;
#define POLYA_DELTA 0
#define POLYA_EPSILON 0
#define THRESHOLD 0.01

/**
 * Return an array of size training_size containing values distributed according to N(0,1)
*/
double* generate_normal_sequence() {
    double* normal_sequence = (double*) malloc(training_size * sizeof(double));
    std::default_random_engine rng;
    rng.seed(31);
    std::normal_distribution<double> distribution(0, 1);
    for(int i = 0; i < training_size; i++) {
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

void compute_error_matrix(double* error_matrix, unsigned int levels, unsigned int bit_rate) {
  for(int i = 0; i < levels; i++) {
    for(int j = 0; j < levels; j++) {
      error_matrix[j + i * levels] = polya_urn_error(j, i, bit_rate);
    }
  }
}

void s_nnc(double* training_sequence, double* codebook, int levels, double* error_matrix,
    double* cc_sums, unsigned int* cc_cardinality) {
  double min = __FLT_MAX__;
  int min_index = -1;
  double sum = 0;
  double c = 0;
  for(int i = 0; i < training_size; i++) {
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
    // For Centroid Condition
    cc_cardinality[min_index]++; // update count
    cc_sums[min_index] += target; // running sum
    sum = 0;
    min_index = -1;
    min = __FLT_MAX__;
  }
}

void nnc(unsigned int* cells, double* training_sequence, double* codebook, int levels, double* error_matrix,
    double* cc_sums, unsigned int* cc_cardinality) {
  double min = __FLT_MAX__;
  int min_index = -1;
  double sum = 0;
  double c = 0;
  for(int i = 0; i < training_size; i++) {
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
  for(int i = 0; i < training_size; i++) {
    for(int j = 0; j < levels; j++) {
      double y = error_matrix[j + levels*cells[i]] * (training_sequence[i] - codebook[j]) * (training_sequence[i] - codebook[j]) - c;
      double t = d + y;
      c = (t - d) - y;
      d = t;
    }
  }
  return d / training_size;
}

/**
 * Splitting technique:
 * - A study of vector quantization for noisy channels, pg. 806 B.
 * - An Algorithm for Vector Quantizer Design pg. 89
*/
double* split(double* training_sequence, int levels, double* error_matrix,
    double* cc_cell_sums, unsigned int* cc_cell_cardinality) {
  double delta = 0.001;
  double* temp = NULL;
  double* s_codebook = (double*) malloc(sizeof(double) * levels);
  double* codebook = (double*) malloc(sizeof(double) * levels);
  // Compute centroid of training sequence
  double sum = 0;
  for(int i = 0; i < training_size; i++)
    sum += training_sequence[i];
  codebook[0] = sum / training_size;
  // Splitting loop
  unsigned int rate = 0;
  unsigned int s_levels = 1;
  while(s_levels < levels) {
    for(int i = 0; i < s_levels; i++) {
      s_codebook[2*i] = codebook[i] - delta;
      s_codebook[2*i+1] = codebook[i] + delta;
    }
    temp = codebook;
    codebook = s_codebook;
    s_codebook = temp;
    s_levels <<= 1;
    rate++;
    memset(cc_cell_sums, 0, sizeof(double) * s_levels);
    memset(cc_cell_cardinality, 0, sizeof(unsigned int) * s_levels);
    compute_error_matrix(error_matrix, s_levels, rate);
    s_nnc(training_sequence, codebook, s_levels, error_matrix, cc_cell_sums, cc_cell_cardinality);
    cc(s_levels, error_matrix, cc_cell_sums, cc_cell_cardinality, codebook);
  }
  free(s_codebook);
  return codebook;
}

void cosq(double* training_sequence) {
  int levels = 1 << rate;
  double dist_curr = 0, dist_prev = DBL_MAX;
  double* error_matrix = (double*) malloc(sizeof(double) * levels * levels);
  unsigned int* cells = (unsigned int*) malloc(sizeof(unsigned int) * training_size);
  double* cc_cell_sums = (double*) malloc(sizeof(double) * levels);
  unsigned int* cc_cell_cardinality = (unsigned int*) malloc(sizeof(unsigned int) * levels);
  double* codebook = split(training_sequence, levels, error_matrix, cc_cell_sums, cc_cell_cardinality);
  compute_error_matrix(error_matrix, levels, rate);
  // Lloyd Iteration
  while(true) {
    memset(cc_cell_sums, 0, sizeof(double) * levels);
    memset(cc_cell_cardinality, 0, sizeof(unsigned int) * levels);
    nnc(cells, training_sequence, codebook, levels, error_matrix, cc_cell_sums, cc_cell_cardinality);
    cc(levels, error_matrix, cc_cell_sums, cc_cell_cardinality, codebook);
    dist_curr = distortion(levels, training_sequence, error_matrix, codebook, cells);
    if((dist_prev - dist_curr) / dist_prev < THRESHOLD) {
      break;
    }
    dist_prev = dist_curr;
  }
  free(codebook);
  free(error_matrix);
}

int main(int argc, char** argv) {
  rate = atoi(argv[1]);
  training_size = atoi(argv[2]);
  printf("Training %d-bit normal quantizer for polya channel delta %f, epsilon %f\n", rate, POLYA_DELTA, POLYA_EPSILON);
  double* normal = generate_normal_sequence();
  std::chrono::_V2::system_clock::time_point start, end;
  std::chrono::seconds exec_time;
  int sum = 0;
  for(int i = 0; i < 11; i++) {
    start = std::chrono::high_resolution_clock::now();
    cosq(normal);
    end = std::chrono::high_resolution_clock::now();
    exec_time = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    if(i == 0) {
      std::cout << "Warm-up time is " << exec_time.count() << "s." << std::endl;
    } else {
      sum += exec_time.count();
    }
  }
  std::cout << "The average is " << (float) sum / 10 << "s." << std::endl;
  free(normal);
  return 0;
}
