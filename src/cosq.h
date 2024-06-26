#include "util/cuda_util.h"

#define POLYA_EPSILON 0
#define POLYA_DELTA 0
#define THRESHOLD 0.01
#define WARP_SIZE 32

class COSQ;
class Device {
    friend class COSQ;
    public:
        Device(COSQ* cosq);
        ~Device();
    private:
        // General
        double* training_sequence;
        double* error_matrix;
        double* q_points;
        // NNC
        dim3 nnc_ge32_grid_size;
        dim3 nnc_ge32_block_size;
        dim3 nnc_lt32_grid_size;
        dim3 nnc_lt32_block_size;
        unsigned int nnc_smem_size;
        unsigned int* q_cells;
        // CC
        dim3 cc_grid_size;
        dim3 cc_block_size;
        unsigned int* cc_cardinality;
        double* cc_cell_sums;
        // Distortion
        dim3 dist_grid_size;
        dim3 dist_block_size;
        unsigned int dist_smem_size;
        double* reduction_sums;
};
class COSQ {
    friend class Device;
    public:
        COSQ(double* training_sequence, const unsigned int* training_size, const unsigned int* bit_rate);
        ~COSQ();
        double* train();
    private:
        Device* device;
        double* training_sequence;
        unsigned int training_size;
        unsigned int levels;
        unsigned int bit_rate;
        double* error_matrix;
        double* q_points;
        void cc_lt32(double* cc_sums, unsigned int* cc_cardinality);
        inline double polya_urn_error(int j, int i, int num_bits);
        void compute_error_matrix(double* error_matrix, unsigned int levels, unsigned int bit_rate);
        double* cosq_lt32();
        double* cosq_ge32();
};