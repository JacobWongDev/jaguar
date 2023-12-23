#include "pgm_util.h"
#include "logger.h"
#include <stdio.h>

pgm_image* load_image(const char* loc) {
    int width = 0;
    int height = 0;
    int intensity_scale = 0;
    pgm_image* image = NULL;
    FILE* pgm = fopen(loc, "r");
    if(pgm == NULL) {
        logger_send("Couldn't open image file!", ERROR);
        return NULL;
    }
    // Read PGM header
    char format[2];
    fscanf(pgm, "%2s", format);
    if (format[0] != 'P' || format[1] != '5') {
        logger_send("Invalid PGM format!: Header incorrect!\n", ERROR);
        fclose(pgm);
        return NULL;
    }
    // Skip comment line, if exists
    //TODO
    // Read width and height of image
    fscanf(pgm, "%d %d", &width, &height);
    // Verify width and height
    fprintf(stdout, "%c %c\n", format[0], format[1]);
    fflush(stdout);
    fprintf(stdout, "%d %d\n", width, height);
    fflush(stdout);
    if((width % 8 != 0 || width == 0) || (height % 8 != 0 || height == 0)) {
        logger_send("Invalid PGM format!: Width and height must be non-zero multiples of 8!\n", ERROR);
        fclose(pgm);
        return NULL;
    }
    // Read intensity scale
    fscanf(pgm, "%d", &intensity_scale);
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

void free_image(pgm_image* image) {
    free(image->intensity);
    image->intensity = NULL;
    free(image);
    image = NULL;
}