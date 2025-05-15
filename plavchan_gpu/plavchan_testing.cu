#include "./plavchan_periodogram.cu"
#include <stdio.h>

int main() {
    // Constructing test cases
    int N_OBJS = 100;
    int N_PDS = 100000;
    int N_POINTS = 500;

    int MAX_TIME = 10;

    Array2D mags, times;
    // define times as random times between 0 and MAX_TIME
    times.dim1 = N_OBJS;
    times.dim2 = (size_t*)malloc(N_OBJS * sizeof(size_t));
    times.array = (float**)malloc(N_OBJS * sizeof(float*));
    for (size_t i = 0; i < N_OBJS; i++) {
        times.dim2[i] = N_POINTS;
        times.array[i] = (float*)malloc(N_POINTS * sizeof(float));
        for (size_t j = 0; j < N_POINTS; j++) {
            times.array[i][j] = ((float)rand() / RAND_MAX) * MAX_TIME;
        }
    }

    // define mags as the sine of the times at an increasing period
    mags.dim1 = N_OBJS;
    mags.dim2 = (size_t*)malloc(N_OBJS * sizeof(size_t));
    mags.array = (float**)malloc(N_OBJS * sizeof(float*));
    for (size_t i = 0; i < N_OBJS; i++) {
        mags.dim2[i] = N_POINTS;
        mags.array[i] = (float*)malloc(N_POINTS * sizeof(float));
        float period = (float)(i + 1) / N_OBJS;
        for (size_t j = 0; j < N_POINTS; j++) {
            mags.array[i][j] = sinf(times.array[i][j] * period);
        }
    }
    
    Array1D pds;
    pds.dim1 = N_PDS;
    pds.array = (float*)malloc(sizeof(float)*N_PDS);
    for (int i = 0; i < N_PDS; i++) {
        pds.array[i] = ((float)(i+1) / N_PDS) * MAX_TIME;
    }

    // Call the function
    Array2D periodogram = plavchan_periodogram(mags, times, pds, 0.1f);

    return 0;
}