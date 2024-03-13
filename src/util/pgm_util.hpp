#include <iostream>

typedef struct {
    unsigned int height;
    unsigned int width;
    unsigned char* intensity;
} pgm_image;

float* MallocPlaneFloat(unsigned int width, unsigned int height, unsigned int *pStepBytes);

void copy_plane(unsigned char* src, unsigned int stride_src, float* result, unsigned int stride_result, unsigned int image_width, unsigned int image_height);

pgm_image* load_image(const char* loc);

void save_image(int height, int width, float* values);

void free_image(pgm_image* image);
