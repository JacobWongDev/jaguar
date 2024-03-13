#pragma once

/**
 * Only used in Simple COSQ
*/
#define TRAINING_SIZE 100000

/**
*  The dimension of pixels block
*/
#define BLOCK_SIZE 8

/**
*  Square of dimension of pixels block
*/
#define BLOCK_SIZE2 64

/**
 * Maximum Image Height
*/
#define MAX_IMAGE_HEIGHT 1080

/**
 * Maximum Image Width
*/
#define MAX_IMAGE_WIDTH 1920

/**
 * Number of BLOCK_SIZE * BLOCK_SIZE blocks per image
*/
#define MAX_BLOCK_COUNT (MAX_IMAGE_HEIGHT * MAX_IMAGE_WIDTH) / BLOCK_SIZE2

/**
 * Maximum bit allocation
*/
#define MAX_BIT_ALLOCATION 8

/**
 * Maximum codebook size for COSQ
*/
#define MAX_CODEBOOK_SIZE 256

/**
*  This macro states that __mul24 operation is performed faster that traditional
*  multiplication for two integers on CUDA. Please undefine if it appears to be
*  wrong on your system
*/
#define __MUL24_FASTER_THAN_ASTERIX

/**
*  Wrapper to the fastest integer multiplication function on CUDA
*/
#ifdef __MUL24_FASTER_THAN_ASTERIX
#define FMUL(x, y) (__mul24(x, y))
#else
#define FMUL(x, y) ((x) * (y))
#endif