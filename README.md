# CrNeuralNet

A lightweight C++ implementation of a neural network library with matrix operations, layer functionality, and automatic differentiation for backpropagation.

## Features

- **Matrix Operations**: Addition, subtraction, multiplication, transpose, Hadamard product
- **Activation Functions**: ReLU, Softmax, and their derivatives
- **Neural Network Layers**: Input, Hidden, and Output layers with forward/backward propagation
- **Weight Initialization**: Xavier, He, Random, and Zero initialization methods
- **Training**: Complete training loop with gradient descent and accuracy tracking
- **Modern C++**: Uses std::vector for memory management, operator overloading, and smart pointers

## Project Structure

```
CrNeuralNet/
├── include/
│   ├── Matrix.hpp      # Matrix class definition
│   ├── Layer.hpp       # Layer hierarchy (LayerBase, Layer, HiddenLayer, OutputLayer)
│   ├── Network.hpp     # Network class definition
│   ├── Dataset.hpp     # Dataset class for training data
│   └── InitType.hpp    # Weight initialization types
├── src/
│   ├── Matrix.cpp      # Matrix implementation
│   ├── Layer.cpp       # Layer implementation
│   ├── Network.cpp     # Network implementation
│   └── Dataset.cpp     # Dataset implementation
├── build/              # Object files directory (created during compilation)
├── test_simple.cpp     # Example test file
├── Makefile           # Build automation
└── README.md
```

## Compilation

### Using Makefile (Recommended)

The project includes a Makefile for easy compilation and management:

```bash
# Compile the entire project
make

# Clean build files
make clean

# Rebuild everything (clean + build)
make rebuild

# Build and run the test
make run

# Show available commands
make help
```

### Manual Compilation

If you prefer manual compilation:

```bash
# Create build directory
mkdir -p build

# Compile object files
g++ -std=c++17 -I include -c src/Matrix.cpp -o build/Matrix.o
g++ -std=c++17 -I include -c src/Dataset.cpp -o build/Dataset.o
g++ -std=c++17 -I include -c src/Layer.cpp -o build/Layer.o
g++ -std=c++17 -I include -c src/Network.cpp -o build/Network.o

# Link executable
g++ -std=c++17 -I include -o test_simple test_simple.cpp build/*.o -framework Accelerate
```

## Usage

### Basic Neural Network Example

```cpp
#include "include/Network.hpp"
#include "include/Dataset.hpp"
#include "include/Matrix.hpp"
#include <vector>

int main() {
    // Create training data (XOR problem)
    std::vector<Matrix> inputs;
    std::vector<size_t> outputs;
    
    inputs.push_back(Matrix(2, 1, {0.0, 0.0}));
    outputs.push_back(0);
    inputs.push_back(Matrix(2, 1, {0.0, 1.0}));
    outputs.push_back(1);
    inputs.push_back(Matrix(2, 1, {1.0, 0.0}));
    outputs.push_back(1);
    inputs.push_back(Matrix(2, 1, {1.0, 1.0}));
    outputs.push_back(0);
    
    // Create dataset
    Dataset dataset(inputs, outputs);
    
    // Create network: 2 input -> 4 hidden -> 2 output
    Network network({2, 4, 2}, InitType::He, 0.1);
    
    // Train the network
    network.train(dataset, 100);
    
    return 0;
}
```

### Matrix Operations

```cpp
#include "include/Matrix.hpp"

// Create matrices
Matrix m1(2, 3, {1, 2, 3, 4, 5, 6});
Matrix m2(3, 2, {1, 2, 3, 4, 5, 6});

// Matrix operations
Matrix result = m1 * m2;           // Matrix multiplication
Matrix sum = m1 + m2.transpose();  // Addition with transpose
Matrix relu = m1.relu();           // ReLU activation
Matrix softmax = m1.softmax();    // Softmax activation

// Print matrix
result.print();
```

## Weight Initialization

The network supports different weight initialization methods:

- `InitType::Zero` - Initialize all weights to zero
- `InitType::Rand` - Random initialization
- `InitType::He` - He initialization (recommended for ReLU)
- `InitType::Xavier` - Xavier/Glorot initialization

## Requirements

- **Compiler**: C++17 compatible compiler (g++, clang++)
- **Platform**: macOS (uses Accelerate framework for optimized operations)
- **Dependencies**: None (uses only standard library)

## Performance

The implementation uses:
- **Accelerate Framework**: For optimized matrix operations on macOS
- **Efficient Memory Management**: std::vector with pre-allocated sizes
- **Smart Pointers**: std::unique_ptr for automatic memory management

## License

This project is open source and available under the MIT License.
