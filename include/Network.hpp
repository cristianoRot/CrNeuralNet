// network.hpp

#pragma once
#include "Layer.hpp"
#include "Matrix.hpp"
#include "Dataset.hpp"
#include <vector>
#include <tuple>
#include <memory>

class Network 
{
    private:
        Matrix input_layer;
        std::vector<HiddenLayer> layers;
        std::unique_ptr<OutputLayer> output_layer;

        double learning_rate;

    public:
        Network(std::vector<size_t> layer_sizes, double learning_rate);

        const Matrix& get_output() const;

        void train(Dataset& dataset, size_t epochs);
        void forward(const Matrix& input);
        void backprop(const Matrix& y);
        void step(double learning_rate);

        Matrix loss(const Matrix& y);
};
