
void center_pixels(float* pixels, int stride, int width, int height);

/**
 * @brief Create a COSQ based on provided training images
 *
 */
int train(const char* dir_name, const char* channel_name);