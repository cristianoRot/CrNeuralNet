// network.cpp

#include "Network.hpp"

Network::Network(std::vector<size_t> layer_sizes, double learning_rate)
{
    if (layer_sizes.size() < 2)
    {
        throw std::invalid_argument("Error: Network must have at least 2 layers");
    }

    this->learning_rate = learning_rate;

    input_layer = Matrix(layer_sizes[0], 1);

    const Matrix* prev_A = &input_layer;
    Matrix* prev_dA = nullptr;
    
    for (size_t i = 1; i < layer_sizes.size() - 1; i++)
    {
        layers.push_back(HiddenLayer(layer_sizes[i - 1], 
                                    layer_sizes[i],
                                    *prev_A,
                                    prev_dA));

        HiddenLayer& last_layer = layers.back();
        
        prev_A = &last_layer.getA();
        prev_dA = &last_layer.get_dA();
    }

    size_t vector_size = layer_sizes.size();
    output_layer = std::make_unique<OutputLayer>(
        layer_sizes[vector_size - 2],
        layer_sizes[vector_size - 1],
        *prev_A,
        prev_dA
    );
}

const Matrix& Network::get_output() const { return output_layer->getA(); }

void Network::train(Dataset& dataset, size_t epochs)
{
    for (size_t e = 0; e < epochs; e++)
    {
        dataset.shuffle();

        for (size_t i = 0; i < dataset.size(); i++)
        {
            forward(dataset.get_input(i));
            backprop(dataset.get_output(i));
            step(learning_rate);
        }
    }
}

void Network::forward(const Matrix& input)
{
    input_layer = input; // Set input layer

    for (size_t i = 0; i < layers.size(); i++)
    {
        layers[i].forward(); 
    }

    output_layer->forward();
}

void Network::backprop(const Matrix& y)
{
    output_layer->set_dZ(loss(y));
    output_layer->backprop();

    for (size_t i = layers.size(); i-- > 0; )
    {
        layers[i].backprop();
    }
}

Matrix Network::loss(const Matrix& y)
{
    return output_layer->getA() - y;
}

void Network::step(double learning_rate)
{
    for (size_t i = 0; i < layers.size(); i++)
    {
        layers[i].step(learning_rate);
    }

    output_layer->step(learning_rate);
}