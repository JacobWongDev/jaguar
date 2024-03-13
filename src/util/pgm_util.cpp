#include "pgm_util.hpp"
#include "logger.hpp"
#include <stdio.h>
#include <math.h>

void copy_plane(unsigned char* src, unsigned int src_pitch, float* result, unsigned int dst_pitch, unsigned int image_width, unsigned int image_height) {
    for (int i = 0; i < image_height; i++) {
        for (int j = 0; j < image_width; j++) {
            result[i * dst_pitch + j] = (float) src[i * src_pitch + j];
        }
    }
}

/**
**************************************************************************
*  Memory allocator, returns aligned format frame with 32bpp float pixels.
*
* \param width          [IN] - Width of image buffer to be allocated
* \param height         [IN] - Height of image buffer to be allocated
* \param pStepBytes     [OUT] - Step between two sequential rows
*
* \return Pointer to the created plane
*/
float* MallocPlaneFloat(unsigned int width, unsigned int height, unsigned int* pitch) {
    float *ptr;
    *pitch = width * sizeof(float);
    ptr = (float *)malloc(*pitch * height);
    *pitch = *pitch / sizeof(float);
    return ptr;
}

pgm_image* load_image(const char* loc) {
    int width = 0;
    int height = 0;
    int intensity_scale = 0;
    int MAXLENGTH = 1000;
    char line[MAXLENGTH];
    pgm_image* image = NULL;
    FILE* pgm = fopen(loc, "r");
    if(pgm == NULL) {
        logger_send("Couldn't open image file!", ERROR);
        return NULL;
    }
    // Read PGM header
    do {
        fgets(line, MAXLENGTH, pgm);
    } while(line[0]=='#' || line[0]=='\n');
    if(line[0] != 'P' || line[1] != '5') {
        logger_send("Invalid PGM format!: Header incorrect!\n", ERROR);
        fclose(pgm);
        return NULL;
    }
    // Read width and height of image
    do {
        fgets(line, MAXLENGTH, pgm);
    } while(line[0]=='#' || line[0]=='\n');
    sscanf(line, "%d %d", &width, &height);
    // Verify width and height
    if((width % 8 != 0 || width == 0) || (height % 8 != 0 || height == 0)) {
        logger_send("Invalid PGM format!: Width and height must be non-zero multiples of 8!\n", ERROR);
        fclose(pgm);
        return NULL;
    }
    // Read intensity scale
    do {
        fgets(line, MAXLENGTH, pgm);
    } while(line[0]=='#' || line[0]=='\n');
    sscanf(line, "%d", &intensity_scale);
    if(intensity_scale != 255) {
        logger_send("Invalid PGM format!: Maximum grayscale values must be 255!\n", ERROR);
        fclose(pgm);
        return NULL;
    }
    // Allocate memory for image data
    image = (pgm_image*) malloc(sizeof(pgm_image));
    if (image == NULL) {
        logger_send("Memory allocation error: image!\n", ERROR);
        fclose(pgm);
        return NULL;
    }
    image->width = width;
    image->height = height;
    // Read image data
    image->intensity = (unsigned char*) malloc(sizeof(unsigned char) * width * height);
    fread(image->intensity, sizeof(unsigned char), width * height, pgm);
    if (image->intensity == NULL) {
        logger_send("Memory allocation error: image->intensity!\n", ERROR);
        fclose(pgm);
        return NULL;
    }
    // Close the file
    fclose(pgm);
    return image;
}

void save_image(int height, int width, float* values) {
    int MAXLENGTH = 1000;
    char line[MAXLENGTH];
    FILE* pgm = fopen("Lena_test.pgm", "w");
    if(pgm == NULL) {
        logger_send("Couldn't open image file!", ERROR);
    }
    fprintf(pgm, "P5\n");
    fprintf(pgm, "%d %d\n", width, height);
    fprintf(pgm, "255\n");

    for(int i = 0; i < width * height; i++) {
        fputc((int) values[i], pgm);
    }
    fclose(pgm);
}

void free_image(pgm_image* image) {
    free(image->intensity);
    image->intensity = NULL;
    free(image);
    image = NULL;
}