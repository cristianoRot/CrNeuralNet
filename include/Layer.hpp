// layer.hpp

#pragma once
#include "Matrix.hpp"

enum class Activation {
    None,
    ReLU,
    Softmax
};

class Layer 
{
    private:   
        Matrix A;
        Matrix b;
        Matrix W;
        Matrix Z;

        Matrix dA;
        Matrix db;
        Matrix dW;
        Matrix dZ;

    public:
        Layer(size_t input_size, size_t output_size);

        void forward(const Matrix& prev_A, Activation activation);
        void backprop(const Matrix& prev_A, Matrix& prev_dA);

        const Matrix& getA();
        void set_dA(const Matrix& g);
};
