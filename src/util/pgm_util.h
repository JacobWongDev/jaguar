#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif
    typedef struct {
        unsigned int height;
        unsigned int width;
        unsigned char* intensity;
    } pgm_image;

    float* MallocPlaneFloat(int width, int height, int *pStepBytes);

    void copy_plane(unsigned char* src, int stride_src, float* result, int stride_result, int image_width, int image_height);

    pgm_image* load_image(const char* loc);

    void free_image(pgm_image* image);
#ifdef __cplusplus
}
#endif