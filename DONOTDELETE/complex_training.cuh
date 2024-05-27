
void cuda_dct(
        float* local_matrix,
        float* device_matrix_b,
        float* device_matrix_a,
        unsigned int image_width,
        unsigned int image_height,
        unsigned int device_matrix_pitch,
        unsigned int local_matrix_pitch);

void cuda_cosq(float* matrix, float* device_plane_result, float* device_plane_src, unsigned int image_width, unsigned int image_height, unsigned int pitch, unsigned int matrix_stride);

void center_pixels(float* pixels, int stride, int width, int height);

/**
 * @brief Create a COSQ based on provided training images
 *
 */
int train(const char* dir_name, const char* channel_name);