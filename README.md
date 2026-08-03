# MLP for Handwritten Digits

This project is a practice implementation of a neural network from scratch in C. It generally achieves an accuracy of ~97% within a single epoch. The implementation doesn't attempt to adopt scalability, although the dimension sizes of the parameters are available for modification.

This specific configuration adopts a 784 -> 128 -> 64 -> 10 architecture, which has been most commonly used for handwritten digit classification.

Training is accelerated with OpenCL: the forward pass, backpropagation and the SGD update are executed as kernels on an OpenCL device, and the parameters and activations remain in device memory for the duration of training. The kernels are compiled at runtime from `kernels.cl`.

## Project Structure

```
mlp-mnist/
├── README.md
├── .gitignore
└── src/
    ├── main.c          # Data loading, the training loop and validation
    ├── cl_ops.c        # OpenCL host code: device setup, buffers, kernel launches
    ├── cl_ops.h        # Interface to the OpenCL host code
    ├── kernels.cl      # The OpenCL kernels: matmul, ReLU, softmax, SGD update
    ├── makefile        # Provides make functionality for this project
    ├── get-data.sh     # Downloads the MNIST dataset; invoked by "make data"
    ├── nn              # Compiled binary; created by "make" and not tracked
    └── data/           # MNIST idx files; created by "make data" and not tracked
```

## Requirements

This project requires an installed C compiler, `make`, `curl`, and `gzip`. It additionally requires OpenCL: the headers and library for compilation, and a working OpenCL runtime for a device at execution. Any OpenCL device is sufficient ()a GPU is the expectation, but a CPU implementation such as PoCL also works); additionally, there is no plain-C fallback, so the program will not run without one. Its validated OS is Linux, but it should also compile and run on Windows and MacOS.

## Usage

To train and validate the network, firstly enter `src/` by running `cd src`, to import the MNIST handwritten digit dataset run `make data`, then compile the code using `make` and to train/validate the neural network run `make run`. Note that the binary loads `kernels.cl` from the working directory at startup, so it should be run from within `src/`, which `make run` already does.
