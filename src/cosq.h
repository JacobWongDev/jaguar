#include "util/cuda_util.h"

#define POLYA_EPSILON 0
#define POLYA_DELTA 0
#define THRESHOLD 0.01
#define WARP_SIZE 32

/**
 * @brief Polya Urn Channel Model. Returns probability
 * of sending i and receiving j.
 *
 * @param j Received codebook index
 * @param i Sent codebook index
 * @param bit_rate bit rate for the quantizer
 * @return probability
 */
inline double polya_urn_error(int j, int i, int bit_rate);

/**
 * Compute the Channel Transition Matrix (ctm).
 *
 * @param ctm pointer to array which compute_ctm will write to.
 * @param levels dictates size of matrix, which is <levels> x <levels>
 * @param bit_rate bit rate for the quantizer
 */
void compute_ctm(double* ctm, unsigned int levels, unsigned int bit_rate);

class COSQ;
class Device {
    friend class COSQ;
    friend class Split;
    public:
        Device(COSQ* cosq);
        ~Device();
    private:
        // General
        double* training_sequence;
        double* ctm;
        double* q_points;
        // NNC
        dim3 nnc_ge5_block_size;
        dim3 nnc_ge5_grid_size;
        unsigned int nnc_ge5_smem_size;
        dim3 nnc_lt5_block_size;
        dim3 nnc_lt5_grid_size;
        unsigned int* q_cells;
        // CC
        dim3 cc_gather_grid_size;
        dim3 cc_gather_block_size;
        unsigned int cc_gather_smem_size;
        dim3 cc_ge5_grid_size;
        dim3 cc_ge5_block_size;
        unsigned int cc_ge5_smem_size;
        dim3 cc_le5_grid_size;
        dim3 cc_le5_block_size;
        unsigned int* cc_cardinality;
        double* cc_cell_sums;
        // Distortion
        dim3 dist_grid_size;
        dim3 dist_block_size;
        unsigned int dist_smem_size;
        //unsigned int dist_smem_size;
        double* reduction_sums;
};
class Split {
    public:
        Split(COSQ* cosq, Device* device);
        /**
         * @brief Executes splitting technique for bit rates less than 5.
         * COSQ::q_points will be written to.
         */
        void split_lt5();
        /**
         * @brief Executes splitting technique for bit rates greater or equal to 5.
         * COSQ::q_points will be written to.
         */
        void split_ge5();
    private:
        static constexpr double delta = 0.001;
        dim3 nnc_ge5_block_size;
        dim3 nnc_ge5_grid_size;
        unsigned int nnc_ge5_smem_size;
        dim3 nnc_lt5_block_size;
        dim3 nnc_lt5_grid_size;
        // CC
        dim3 cc_gather_grid_size;
        dim3 cc_gather_block_size;
        unsigned int cc_gather_smem_size;
        dim3 cc_ge5_grid_size;
        dim3 cc_ge5_block_size;
        unsigned int cc_ge5_smem_size;
        dim3 cc_le5_grid_size;
        dim3 cc_le5_block_size;
        COSQ* cosq;
        Device* device;
};
class COSQ {
    friend class Device;
    friend class Split;
    public:
        COSQ(double* training_sequence, const unsigned int* training_size, const unsigned int* bit_rate);
        ~COSQ();
        /**
         * @brief Run COSQ algorithm and return codebook to target_q_points.
         */
        void train(double* target_q_points);
    private:
        Device* device;
        double* training_sequence;
        unsigned int training_size;
        unsigned int levels;
        unsigned int bit_rate;
        double* ctm;
        double* q_points;
        /**
         * @brief Executes COSQ algorithm for bit rates less than 5.
         * target_q_points will be written to.
         */
        void cosq_lt5(double* target_q_points);
        /**
         * @brief Executes COSQ algorithm for bit rates greater or equal to 5.
         * target_q_points will be written to.
         */
        void cosq_ge5(double* target_q_points);
};