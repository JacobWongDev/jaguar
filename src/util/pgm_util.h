#include <stdio.h>

typedef struct {
    unsigned int height;
    unsigned int width;
    unsigned char* intensity;
} pgm_image;

pgm_image* load_image(const char* loc);

void free_image(pgm_image* image);