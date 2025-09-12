// layer.cpp

#include "Layer.hpp"

Layer::Layer(size_t input_size, size_t output_size)
{
    A = Matrix(output_size, 1);
    b = Matrix(output_size, 1);
    W = Matrix(output_size, input_size);
    Z = Matrix(output_size, 1);

    dA = Matrix(output_size, 1);
    db = Matrix(output_size, 1);
    dW = Matrix(output_size, input_size);
    dZ = Matrix(output_size, 1);
}

const Matrix& Layer::getA() { return A; }
void Layer::set_dA(const Matrix& g) { dA = g; }

void Layer::forward(const Matrix& prev_A, Activation activation)
{
    Z = (W * prev_A) + b;

    switch (activation)
    {
        case Activation::ReLU:
            A = Z.relu();
            break;
        case Activation::Softmax:
            A = Z.softmax();
            break;
        default:
            break;
    }
}

void Layer::backprop(const Matrix& prev_A, Matrix& prev_dA)
{
    dZ = dA.hadamard(Z.drelu());
    dW = dZ * prev_A.transpose();
    prev_dA = W.transpose() * dZ;
    db = dZ;
}
