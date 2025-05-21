#include <Python.h>
#include <float.h>
#include "./plavchan_periodogram.cu"

// compile command: 
// nvcc -c plavchan.cu -o plavchan.exe --compiler-options -fPIC --std=c++14 -I/home/mpaz/anaconda3/envs/period/include/python3.9 -I/home/mpaz/anaconda3/envs/period/include -I/usr/local/cuda/include


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
