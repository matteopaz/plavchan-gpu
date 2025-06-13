#include <Python.h>
#include <float.h>
// #include "./plavchan_periodogram.cu"

// START PLAVCHAN_PERIODOGRAM.CU
// IMPORT NOT INCLUDED, IT BREAKS PYPI BUILD

#include <float.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
// #include <chrono>

const int nBlocks = 256;
const int nThreads = 512;

typedef struct {
    float* array;
    size_t dim1;
} Array1D;

typedef struct {
    float** array;
    size_t dim1;
    size_t* dim2;
} Array2D;

__device__ void logArr1D(Array1D* arr) {
    for (size_t i = 0; i < arr->dim1; i++) {
        printf("%f ", arr->array[i]);
    }
    printf("\n");
}

__device__ float getMax(Array1D* arr) {
    float max = -FLT_MAX;
    for (int i = 0; i < arr->dim1; i++) {
        if (arr->array[i] > max) {
            max = arr->array[i];
        }
    }
    return max;
}

__device__ float getMin(Array1D* arr) {
    float min = FLT_MAX;
    for (int i = 0; i < arr->dim1; i++) {
        if (arr->array[i] < min) {
            min = arr->array[i];
        }
    }
    return min;
}

__device__ void swap(float* a, float* b)
{
    float t = *a;
    *a = *b;
    *b = t;
}

__device__ int partition(Array1D* arr, Array1D* sim, int l, int h)
{
    float x = arr->array[h];
    int i = (l - 1);
 
    for (int j = l; j <= h - 1; j++) {
        if (arr->array[j] <= x) {
            i++;
            swap(arr->array + i, arr->array + j);
            swap(sim->array + i, sim->array + j);
        }
    }
    swap(&arr->array[i + 1], &arr->array[h]);
    swap(&sim->array[i + 1], &sim->array[h]);
    return (i + 1);
}
 
__device__ void simulSort(Array1D* main, Array1D* sim, float* stack_buf) {   
    if (main->dim1 != sim->dim1) {
        printf("Error: Array dimensions do not match.\n");
        return;
    }

    // initialize top of stack
    int top = -1;
    int l = 0;
    int h = main->dim1 - 1;
 
    // push initial values of l and h to stack
    stack_buf[++top] = l;
    stack_buf[++top] = h;

    // Keep popping from stack while is not empty
    while (top >= 0) {
        // Pop h and l
        h = stack_buf[top--];
        l = stack_buf[top--];
 
        // Set pivot element at its correct position
        // in sorted array
        int p = partition(main, sim, (int)l, (int)h);
 
        // If there are elements on left side of pivot,
        // then push left side to stack
        if (p - 1 > l) {
            stack_buf[++top] = l;
            stack_buf[++top] = p - 1;
        }
 
        // If there are elements on right side of pivot,
        // then push right side to stack
        if (p + 1 < h) {
            stack_buf[++top] = p + 1;
            stack_buf[++top] = h;
        }
    }
}

__device__ void foldLC(Array1D* mag, Array1D* time, float modulus, float* stack_buf) {
    // raise error if modulus is 0
    if (modulus == 0) {
        printf("Error: Modulus cannot be zero.\n");

        for (int i = 0; i < time->dim1; i++) {
            time->array[i] = 0.0;
        }
    }

    // fold the light curve
    for (int i = 0; i < time->dim1; i++) {
        time->array[i] = fmodf(time->array[i], modulus); // time array should already be zeroed
        time->array[i] /= modulus; // normalize to [0,1]
    }

    simulSort(time, mag, stack_buf);
}

__device__ void boxcar_smoothing(Array1D* m, Array1D* t, float width, Array1D* smoothed) {
    /*
    Writes into `smoothed` a boxcar-smoothed equal-length version of magnitudes
    m: value array
    t: timestamp array on [0,1]
    */
    float halfWidth = width / 2;
    size_t N = t->dim1;
    float runningSum = 0;

    size_t l = 0; // left pointer increment
    size_t r = 0; // right pointer increment    

    for (size_t i = 0; i < N; i++) { // Generate a smoothed value for each point in t
        float leftLimitValue = t->array[i] - halfWidth;
        float rightLimitValue = t->array[i] + halfWidth;

        while (l < r && t->array[l] < leftLimitValue) {
            runningSum -= m->array[l];
            l++;
        }

        while (r < N && t->array[r] < rightLimitValue) {
            runningSum += m->array[r];
            r++;
        }

        smoothed->array[i] = runningSum / (r - l + 1); // average
    }
}

__device__ float plavchan_metric(Array1D* mag, Array1D* time, float width, Array1D* smoothed) { 
    boxcar_smoothing(mag, time, width, smoothed);

    float residue = 0;
    for (size_t i = 0; i < mag->dim1; i++) {
        float raw_diff = mag->array[i] - smoothed->array[i];
        residue += raw_diff * raw_diff;
    }

    return residue;
}

__global__ void plavchan_kernel(Array2D* mags, Array2D* times, Array1D* periods, float* width, 
    Array2D* periodogram, int objId, Array2D* folded_mags_buf , Array2D* folded_times_buf, Array2D* smoothed_buf) {
    
    /*
    mags: array of arrays of magnitudes
    time: array of arrays of times, same size as mags
    periods: array of trial periods
    width: fractional width of the boxcar smoothing, between 0 and 1
    periodogram: array of arrays of period values
    objId: the object ID we are working on
    folded_*_buf: buffers for the folded light curves. Dimension (n_concurrent_threads, max_lc_length)
    */

    int periodsPerThread = periods->dim1 / (nBlocks * nThreads) + 1;
    int tId = blockIdx.x * blockDim.x + threadIdx.x;
    int startIdx = tId * periodsPerThread;
    int endIdx = startIdx + periodsPerThread;
    
    if (endIdx > periods->dim1) {
        endIdx = periods->dim1;
    }

    if (startIdx >= periods->dim1) {
        return;
    }

    size_t N = mags->dim2[objId]; // number of points in the light curve, equal to the number of times

    for (size_t i = startIdx; i < endIdx; i++) {
        float period = periods->array[i];

        Array1D folded_mag;
        Array1D folded_time;
        Array1D smoothed;

        folded_mag.array = folded_mags_buf->array[tId];
        folded_time.array = folded_times_buf->array[tId];
        smoothed.array = smoothed_buf->array[tId];
        folded_mag.dim1 = N;
        folded_time.dim1 = N;
        smoothed.dim1 = N;
        
        // copy over the light curve
        for (size_t j = 0; j < N; j++) {
            folded_mag.array[j] = mags->array[objId][j];
            folded_time.array[j] = times->array[objId][j];
        }

        foldLC(&folded_mag, &folded_time, period, smoothed.array); // fold the light curve, uses smooth buffer as stack space for sorting

        float score = plavchan_metric(&folded_mag, &folded_time, *width, &smoothed); // calculate the score
        periodogram->array[objId][i] = N / score; // store the score in the periodogram
    }

    return;
}

// void logArr2D(Array2D* arr) {
//     for (size_t i = 0; i < arr->dim1; i++) {
//         for (size_t j = 0; j < arr->dim2[i]; j++) {
//             printf("%f ", arr->array[i][j]);
//         }
//         printf("\n");
//     }
// }

// void logArr1D(Array1D* arr) {
//     for (size_t i = 0; i < arr->dim1; i++) {
//         printf("%f ", arr->array[i]);
//     }
//     printf("\n");
// }

static Array2D plavchan_periodogram(Array2D mags, Array2D times, Array1D pds, float width) {

    // auto start =  std::chrono::high_resolution_clock::now();

    size_t max_len = 0;
    for (size_t i = 0; i < mags.dim1; i++) {
        max_len = max(max_len, mags.dim2[i]);
    }

    // Allocate and copy MAGS to the GPU
    Array2D* d_mags;
    cudaMalloc(&d_mags, sizeof(Array2D));
    cudaMemcpy(d_mags, &mags, sizeof(Array2D), cudaMemcpyHostToDevice);

    size_t* d_mags_dim2;
    cudaMalloc(&d_mags_dim2, mags.dim1 * sizeof(size_t));
    cudaMemcpy(d_mags_dim2, mags.dim2, mags.dim1 * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_mags->dim2), &d_mags_dim2, sizeof(size_t*), cudaMemcpyHostToDevice);

    float** d_mags_array;
    cudaMalloc(&d_mags_array, mags.dim1 * sizeof(float*));
    for (size_t i = 0; i < mags.dim1; i++) {
        float* d_tempRow;
        cudaMalloc(&d_tempRow, mags.dim2[i] * sizeof(float));
        cudaMemcpy(d_tempRow, mags.array[i], mags.dim2[i] * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mags_array + i, &d_tempRow, sizeof(float*), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(&(d_mags->array), &d_mags_array, sizeof(float**), cudaMemcpyHostToDevice);

    // Allocate and copy TIMES to the GPU
    Array2D* d_times;
    cudaMalloc(&d_times, sizeof(Array2D));
    cudaMemcpy(d_times, &times, sizeof(Array2D), cudaMemcpyHostToDevice);

    size_t* d_times_dim2;
    cudaMalloc(&d_times_dim2, times.dim1 * sizeof(size_t));
    cudaMemcpy(d_times_dim2, times.dim2, times.dim1 * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_times->dim2), &d_times_dim2, sizeof(size_t*), cudaMemcpyHostToDevice);

    float** d_times_array;
    cudaMalloc(&d_times_array, times.dim1 * sizeof(float*));
    for (size_t i = 0; i < times.dim1; i++) {
        float* d_tempRow;
        cudaMalloc(&d_tempRow, times.dim2[i] * sizeof(float));
        cudaMemcpy(d_tempRow, times.array[i], times.dim2[i] * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_times_array + i, &d_tempRow, sizeof(float*), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(&(d_times->array), &d_times_array, sizeof(float**), cudaMemcpyHostToDevice);

    // Allocate and copy TRIAL PERIODS to the GPU
    Array1D* d_periods;
    cudaMalloc(&d_periods, sizeof(Array1D));
    cudaMemcpy(d_periods, &pds, sizeof(Array1D), cudaMemcpyHostToDevice);

    float* d_periods_array;
    cudaMalloc(&d_periods_array, pds.dim1 * sizeof(float));
    cudaMemcpy(d_periods_array, pds.array, pds.dim1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_periods->array), &d_periods_array, sizeof(float*), cudaMemcpyHostToDevice);

    // Allocate and copy WIDTH to the GPU
    float* d_width;
    cudaMalloc(&d_width, sizeof(float));
    cudaMemcpy(d_width, &width, sizeof(float), cudaMemcpyHostToDevice);

    // Create the output variable PERIODGRAM
    Array2D periodogram;
    periodogram.dim1 = mags.dim1;
    periodogram.dim2 = (size_t*)malloc(mags.dim1 * sizeof(size_t));
    for (size_t i = 0; i < mags.dim1; i++) {
        periodogram.dim2[i] = pds.dim1;
    }
    periodogram.array = (float**)malloc(mags.dim1 * sizeof(float*));
    for (size_t i = 0; i < mags.dim1; i++) {
        periodogram.array[i] = (float*)calloc(pds.dim1, sizeof(float));
    }

    // Allocate and copy PERIODGRAM to the GPU
    Array2D* d_periodogram;
    cudaMalloc(&d_periodogram, sizeof(Array2D));
    cudaMemcpy(d_periodogram, &periodogram, sizeof(Array2D), cudaMemcpyHostToDevice);

    size_t* d_periodogram_dim2;
    cudaMalloc(&d_periodogram_dim2, periodogram.dim1 * sizeof(size_t));
    cudaMemcpy(d_periodogram_dim2, periodogram.dim2, periodogram.dim1 * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_periodogram->dim2), &d_periodogram_dim2, sizeof(size_t*), cudaMemcpyHostToDevice);

    float** d_periodogram_array;
    cudaMalloc(&d_periodogram_array, periodogram.dim1 * sizeof(float*));
    for (size_t i = 0; i < periodogram.dim1; i++) {
        float* d_tempRow;
        cudaMalloc(&d_tempRow, periodogram.dim2[i] * sizeof(float));
        cudaMemcpy(d_tempRow, periodogram.array[i], periodogram.dim2[i] * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_periodogram_array + i, &d_tempRow, sizeof(float*), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(&(d_periodogram->array), &d_periodogram_array, sizeof(float**), cudaMemcpyHostToDevice);

    // Allocate buffers for folded light curves
    int n_concurrent_threads = nBlocks * nThreads;

    // MAGS BUF
    Array2D* d_folded_mags_buf;
    cudaMalloc(&d_folded_mags_buf, sizeof(Array2D));

    float* d_folded_mags_buf_dim2;
    cudaMalloc(&d_folded_mags_buf_dim2, n_concurrent_threads * sizeof(size_t));

    float** d_folded_mags_buf_array;
    cudaMalloc(&d_folded_mags_buf_array, n_concurrent_threads*sizeof(float*));
    for (size_t i = 0; i < n_concurrent_threads; i++) {
        float* d_tempRow;
        cudaMalloc(&d_tempRow, max_len*sizeof(float)); // Expanded to longest possible lightcurve
        cudaMemcpy(d_folded_mags_buf_array+i, &d_tempRow, sizeof(float*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_folded_mags_buf_dim2 +i, &max_len, sizeof(size_t), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(&(d_folded_mags_buf->array), &d_folded_mags_buf_array, sizeof(float**), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_folded_mags_buf->dim1), &n_concurrent_threads, sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_folded_mags_buf->dim2), &d_folded_mags_buf_dim2, sizeof(size_t*), cudaMemcpyHostToDevice);


    // TIMES BUF
    Array2D* d_folded_times_buf;
    cudaMalloc(&d_folded_times_buf, sizeof(Array2D));

    float* d_folded_times_buf_dim2;
    cudaMalloc(&d_folded_times_buf_dim2, n_concurrent_threads * sizeof(size_t));

    float** d_folded_times_buf_array;
    cudaMalloc(&d_folded_times_buf_array, n_concurrent_threads*sizeof(float*));
    for (size_t i = 0; i < n_concurrent_threads; i++) {
        float* d_tempRow;
        cudaMalloc(&d_tempRow, max_len*sizeof(float)); // Expanded to longest possible lightcurve
        cudaMemcpy(d_folded_times_buf_array+i, &d_tempRow, sizeof(float*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_folded_times_buf_dim2 +i, &max_len, sizeof(size_t), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(&(d_folded_times_buf->array), &d_folded_times_buf_array, sizeof(float**), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_folded_times_buf->dim1), &n_concurrent_threads, sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_folded_times_buf->dim2), &d_folded_times_buf_dim2, sizeof(size_t*), cudaMemcpyHostToDevice);

    // SMOOTHED BUF
    Array2D* d_folded_smoothed_buf;
    cudaMalloc(&d_folded_smoothed_buf, sizeof(Array2D));

    float* d_folded_smoothed_buf_dim2;
    cudaMalloc(&d_folded_smoothed_buf_dim2, n_concurrent_threads * sizeof(size_t));

    float** d_folded_smoothed_buf_array;
    cudaMalloc(&d_folded_smoothed_buf_array, n_concurrent_threads*sizeof(float*));
    for (size_t i = 0; i < n_concurrent_threads; i++) {
        float* d_tempRow;
        cudaMalloc(&d_tempRow, max_len*sizeof(float)); // Expanded to longest possible lightcurve
        cudaMemcpy(d_folded_smoothed_buf_array+i, &d_tempRow, sizeof(float*), cudaMemcpyHostToDevice);
        cudaMemcpy(d_folded_smoothed_buf_dim2 +i, &max_len, sizeof(size_t), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(&(d_folded_smoothed_buf->array), &d_folded_smoothed_buf_array, sizeof(float**), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_folded_smoothed_buf->dim1), &n_concurrent_threads, sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_folded_smoothed_buf->dim2), &d_folded_smoothed_buf_dim2, sizeof(size_t*), cudaMemcpyHostToDevice);

    // auto end = std::chrono::high_resolution_clock::now();

    // printf("Time taken for memory allocation: %lld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    // fflush(stdout);
    // Launch Kernel
    
    // printf("Launching kernel with %d blocks and %d threads on %d periods \n", nBlocks, nThreads, pds.dim1);

    // size_t freeMem, totalMem;
    // cudaMemGetInfo(&freeMem, &totalMem);
    // freeMem /= 1024*1024; // convert to MB
    // totalMem /= 1024*1024; // convert to MB
    // printf("Free memory: %zu/%zu MB\n", freeMem, totalMem);
    // fflush(stdout);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error before kernel execution: %s\n", cudaGetErrorString(err));
    }

    // start = std::chrono::high_resolution_clock::now();
    for (size_t objId = 0; objId < mags.dim1; objId++) {
        plavchan_kernel<<<nBlocks, nThreads>>>(d_mags, d_times, d_periods, d_width, d_periodogram, objId, 
            d_folded_mags_buf, d_folded_times_buf, d_folded_smoothed_buf);
    }

    err = cudaDeviceSynchronize();

    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error during kernel execution: %s\n", cudaGetErrorString(err));
    }
    // end = std::chrono::high_resolution_clock::now();
    // printf("Kernel execution finished\n");
    // printf("Time taken for computation: %lld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    // fflush(stdout);

    // Copy PERIODGRAM back to host
    for (size_t i = 0; i < periodogram.dim1; i++) {
        float* h_tempRow;
        cudaMemcpy(&h_tempRow, d_periodogram_array + i, sizeof(float*), cudaMemcpyDeviceToHost);
        cudaMemcpy(periodogram.array[i], h_tempRow, periodogram.dim2[i] * sizeof(float), cudaMemcpyDeviceToHost);
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error during copying: %s\n", cudaGetErrorString(err));
    }

    // Free GPU memory
    cudaDeviceReset(); 

    // Z score the periodogram

    for (size_t i = 0; i < periodogram.dim1; i++) {
        float mean = 0;
        float stddev = 0;
        for (size_t j = 0; j < periodogram.dim2[i]; j++) {
            mean += periodogram.array[i][j];
        }
        mean /= periodogram.dim2[i];

        for (size_t j = 0; j < periodogram.dim2[i]; j++) {
            stddev += (periodogram.array[i][j] - mean) * (periodogram.array[i][j] - mean);
        }
        stddev = sqrt(stddev / periodogram.dim2[i]);

        for (size_t j = 0; j < periodogram.dim2[i]; j++) {
            periodogram.array[i][j] = (periodogram.array[i][j] - mean) / stddev;
        }
    }

    // return the proper object
    return periodogram;
}

// START PLAVCHAN.CU

void except(const char* str) { // Throws python exception
    PyErr_SetString(PyExc_TypeError, str);
    exit(1);
}

Array1D parseList(PyObject* list) {
    if (!PyList_Check(list)) {
        except("ERROR: Input must be a list.");
    } 

    Py_ssize_t n_entries = PyList_Size(list);
    float* c_arr = (float*)malloc(n_entries * sizeof(float));
    for (Py_ssize_t i = 0; i < n_entries; i++) {
        PyObject* entry = PyList_GetItem(list, i);
        if (!PyFloat_Check(entry)) {
            except("Entries must be floats");
        }

        c_arr[i] = (float)PyFloat_AsDouble(entry);
    }
    
    Array1D returnval;
    returnval.array = c_arr;
    returnval.dim1 = n_entries;
    return returnval;
}

Array2D parseListofLists(PyObject* lists) {
    if (!PyList_Check(lists)) {
        except("ERROR: Input must be a list.");
    }

    Py_ssize_t n_rows = PyList_Size(lists);
    float** c_arrs = (float**)malloc(n_rows * sizeof(float*));
    size_t* dim2 = (size_t*)malloc(sizeof(size_t) * n_rows);

    for (Py_ssize_t i = 0; i < n_rows; i++) {
        PyObject* innerList = PyList_GetItem(lists, i);
        if (!PyList_Check(innerList)) {
            except("ERROR: Each item in outer list must be a list.");
        }

        Py_ssize_t n_entries = PyList_Size(innerList);
        *(c_arrs + i) = (float*)malloc(n_entries * sizeof(float));
        *(dim2 + i) = n_entries;

        for (Py_ssize_t j = 0; j < n_entries; j++) {
            PyObject* entry = PyList_GetItem(innerList, j);
            if (!PyFloat_Check(entry)) {
                except("Entries must be floats");
            }

            float c_float_entry = (float)PyFloat_AsDouble(entry);
            *(*(c_arrs + i) + j) = c_float_entry;
        }
    }

    Array2D returnval;
    returnval.array = c_arrs;
    returnval.dim1 = n_rows;
    returnval.dim2 = dim2;
    return returnval;
}


static PyObject* PY_plavchan_periodogram(PyObject* self, PyObject* args) {
    PyObject* pymags;
    PyObject* pytimes;
    PyObject* pytrialperiods;
    float width;
    int device_id;
    if (PyArg_ParseTuple(args, "OOOfi", &pymags, &pytimes, &pytrialperiods, &width, &device_id) == 0) {
        return NULL;
    }

    // Parse Python objects into C structures
    Array2D mags = parseListofLists(pymags);
    Array2D times = parseListofLists(pytimes);
    Array1D pds = parseList(pytrialperiods);

    // Safety checks
    if (mags.dim1 != times.dim1) {
        except("Mags and times mismatch in object count.");
        return NULL;
    }
    for (size_t i = 0; i < mags.dim1; i++) {
        if (mags.dim2[i] != times.dim2[i]) {
            char error_message[100];
            snprintf(error_message, sizeof(error_message), 
                     "Mags and times mismatch in entry count in object %zu.", i);
            except(error_message);
            return NULL;
        }
    }


    cudaError_t device_error = cudaSetDevice(device_id);
    if (device_error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(device_error));
        Py_RETURN_NONE;
    }

    Array2D periodogram = plavchan_periodogram(mags, times, pds, width); // actual working step

    // Convert PERIODGRAM to Python object
    PyObject* py_periodogram = PyList_New(periodogram.dim1);
    for (size_t i = 0; i < periodogram.dim1; i++) {
        PyObject* py_tempRow = PyList_New(periodogram.dim2[i]);
        for (size_t j = 0; j < periodogram.dim2[i]; j++) {
            PyObject* py_value = PyFloat_FromDouble(periodogram.array[i][j]);
            PyList_SetItem(py_tempRow, j, py_value);
        }
        PyList_SetItem(py_periodogram, i, py_tempRow);
    }

    return py_periodogram;
}

static PyObject* PY_get_device_count(PyObject* self, PyObject* args) {
    int device_count;
    cudaError_t device_error = cudaGetDeviceCount(&device_count);
    if (device_error != cudaSuccess) {
        printf("CUDA error: %s", cudaGetErrorString(device_error));
        Py_RETURN_NONE;
    }
    return PyLong_FromLong(device_count);
}

// Python integration stuff
static struct PyMethodDef methods[] = {
    {"__cuda__plavchan_pgram", (PyCFunction)PY_plavchan_periodogram, METH_VARARGS, "Compute Plavchan periodogram on GPU"}, 
    {"get_device_count", (PyCFunction)PY_get_device_count, METH_NOARGS, "Get number of CUDA devices"},
    {NULL, NULL, 0, NULL} 
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "plavchan", 
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC PyInit_plavchan(void) { 
    return PyModule_Create(&module);
}
