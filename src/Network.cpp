// network.cpp

#include "Network.hpp"

Network::Network(std::vector<size_t> layer_sizes)
{
    for (size_t i = 1; i < layer_sizes.size(); i++)
    {
        layers.push_back(Layer(layer_sizes[i - 1], layer_sizes[i]));
    }
}

const Matrix& Network::get_output() { return layers[layers.size() - 1].getA(); }

void Network::train(Matrix& input, Matrix& output)
{
    forward(input);
}

void Network::forward(Matrix& input)
{
    Matrix& prev_A = input;

    for (size_t i = 0; i < layers.size(); i++)
    {
        Activation act = i == layers.size() - 1 ? Activation::Softmax : Activation::ReLU;
        
        layers[i].forward(prev_A, act);
        prev_A = layers[i].getA();
    }
}

Matrix& Network::loss(Matrix& output)
{

}