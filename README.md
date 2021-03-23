# Test project for ONNX models in C++

Contains a pythen script which creates a tensorflow/keras model and exports it as ONNX as well as a C++ test program which runs the network.

## Build

```bash
mkdir build
cd build
cmake .. -D CMAKE_PREFIX_PATH="path_to_onnx_folder"
```

The `path_to_onnx_folder` should contain the folders `include` and `lib64`.


## Use

```bash
model/dummy_network.py dummy_network.onnx
build/onnx_cxx_test model/dummy_network.onnx
```
