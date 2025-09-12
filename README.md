# CrNeuralNet

A lightweight C++ implementation of a neural network library with matrix operations and layer functionality.

## Features

- **Matrix Operations**: Addition, subtraction, multiplication, transpose
- **Activation Functions**: ReLU, Softmax, and their derivatives
- **Layer Implementation**: Forward propagation with configurable activation functions
- **Modern C++**: Uses std::vector for memory management and operator overloading

## Project Structure

```
CrNeuralNet/
├── include/
│   ├── Matrix.hpp      # Matrix class definition
│   ├── Layer.hpp       # Layer class definition
│   └── Network.hpp     # Network class definition
├── src/
│   ├── Matrix.cpp      # Matrix implementation
│   ├── Layer.cpp       # Layer implementation
│   └── Network.cpp     # Network implementation
└── README.md
```

## Compilation

```bash
g++ -std=c++17 -I include src/*.cpp -o neural_net
```

## Usage

```cpp
#include "Matrix.hpp"
#include "Layer.hpp"

// Create matrices
Matrix m1(2, 3);
Matrix m2(3, 2);

// Matrix operations
Matrix result = m1 * m2;

// Create a layer
Layer layer(3, 2);
```

## License

This project is open source and available under the MIT License.
