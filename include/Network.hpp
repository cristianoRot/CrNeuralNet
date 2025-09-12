// network.hpp

#pragma once
#include "Layer.hpp"
#include "Matrix.hpp"
#include <vector>

class Network 
{
    private:
        std::vector<Layer> layers;

    public:
        Network(std::vector<size_t> layer_sizes);

        const Matrix& get_output();

        void train(Matrix& input, Matrix& output);
        void forward(Matrix& input);
        void backprop(Matrix& output);

        Matrix& loss(Matrix& output);
};
