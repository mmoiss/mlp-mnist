# MLP for Handwritten Digits

This project is a practice implementation of a neural network from scratch in C. It generally achieves an accuracy of ~95% within a single epoch. The implementation doesn't attempt to adopt scalability, although the dimension size of the parameters is available for modification.

This specific configuration adopts a 784 -> 128 -> 64 -> 10 architecture, which has been most commonly used for handwritten digit classification.

## Project Structure

```
mlp-mnist/
├── README.md
├── .gitignore
└── src/
    ├── main.c          # The entire implementation: data loading, training, validation
    ├── makefile        # Provides make functionality for this project
    ├── get-data.sh     # Downloads the MNIST dataset; invoked by "make data"
    ├── nn              # Compiled binary; created by "make" and not tracked
    └── data/           # MNIST idx files; created by "make data" and not tracked
```

## Requirements

This project requires an installed C compiler, `make`, `curl`, and `gzip`. Its validated OS is Linux, but it should also compile and run on Windows and MacOS.

## Usage

To train and validate the network, firstly enter `src/` by running `cd src`, to import the MNIST handwritten digit dataset run `make data`, then compile the code using `make` and to train/validate the neural network run `make run`.
