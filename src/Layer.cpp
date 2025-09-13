// layer.cpp

#include "Layer.hpp"

// Constructor

Layer::Layer(size_t input_size, size_t output_size, const Matrix& prev_A, Matrix* prev_dA) :
    A(output_size, 1),
    b(output_size, 1), 
    W(output_size, input_size),
    Z(output_size, 1),

    dA(output_size, 1),
    db(output_size, 1),
    dW(output_size, input_size),
    dZ(output_size, 1),

    prev_A(prev_A),
    prev_dA(prev_dA)
{ }

HiddenLayer::HiddenLayer(size_t input_size, size_t output_size, const Matrix& prev_A, Matrix* prev_dA) :
    Layer(input_size, output_size, prev_A, prev_dA) { }

OutputLayer::OutputLayer(size_t input_size, size_t output_size, const Matrix& prev_A, Matrix* prev_dA) :
    Layer(input_size, output_size, prev_A, prev_dA) { }

// Getters and Setters

const Matrix& Layer::getA() const { return A; }
const Matrix& Layer::get_dA() const { return dA; }
Matrix& Layer::get_dA() { return dA; }

void Layer::setA(const Matrix& g) { A = g; }
void Layer::set_dA(const Matrix& g) { dA = g; }

const Matrix& Layer::get_dZ() const { return dZ; }
void Layer::set_dZ(const Matrix& g) { dZ = g; }

void Layer::step(double learning_rate)
{
    W -= (dW * learning_rate);
    b -= (db * learning_rate);
}

// Hidden Layer

void HiddenLayer::forward()
{
    Z = (W * prev_A) + b;
    A = Z.relu();
}

void HiddenLayer::backprop()
{
    dZ = dA.hadamard(Z.drelu());
    dW = dZ * prev_A.transpose();
    db = dZ;

    if (prev_dA != nullptr)
    {
        Matrix temp = W.transpose() * dZ;
        *prev_dA = temp;
    }
}

// Output Layer

void OutputLayer::forward()
{
    Z = (W * prev_A) + b;
    A = Z.softmax();
}

void OutputLayer::backprop()
{
    dW = dZ * prev_A.transpose();
    db = dZ;

    if (prev_dA != nullptr)
    {
        Matrix temp = W.transpose() * dZ;
        *prev_dA = temp;
    }
}