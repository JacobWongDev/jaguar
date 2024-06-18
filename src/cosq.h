#include "util/cuda_util.h"

#define POLYA_EPSILON 0
#define POLYA_DELTA 0
#define THRESHOLD 0.01
#define WARP_SIZE 32

class COSQ {
    private:
        static double* training_sequence;
        static unsigned int training_size;
        static unsigned int levels;
        static unsigned int bit_rate;
        static double* error_matrix;
        static double* q_points;
        class Device {
            public:
                // General
                static double* training_sequence;
                static double* error_matrix;
                static double* q_points;
                // NNC
                static dim3 nnc_grid_size;
                static dim3 nnc_block_size;
                static unsigned int nnc_smem_size;
                static unsigned int* q_cells;
                // CC
                static dim3 cc_grid_size;
                static dim3 cc_block_size;
                static unsigned int* cc_cardinality;
                static double* cc_cell_sums;
                // Distortion
                static dim3 dist_grid_size;
                static dim3 dist_block_size;
                static unsigned int dist_smem_size;
                static double* reduction_sums;
                Device() = delete;
                static void init(double* training_sequence_, const unsigned int* training_size, double* error_matrix_, const unsigned int* levels);
                static void finish();
        };
        static void init(double* training_sequence_, const unsigned int* training_size_);
        static void finish();
        static inline double polya_urn_error(int j, int i, int num_bits);
        static void compute_error_matrix();
        // static void split();
        // static void sim_annealing();
    public:
        COSQ() = delete;
        static double* train(double* training_sequence, const unsigned int* training_size, const unsigned int* bit_rate);

};