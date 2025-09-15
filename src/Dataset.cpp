// dataset.cpp

#include "Dataset.hpp"
#include <random>

Dataset::Dataset(std::vector<Matrix> inputs, std::vector<size_t> outputs)
{
    if (inputs.size() != outputs.size())
    {
        throw std::invalid_argument("Error: Inputs and outputs must have the same size");
    }

    this->inputs = inputs;
    this->outputs = outputs;

    perm_idx.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++)
    {
        perm_idx.push_back(i);
    }
}

const size_t Dataset::size() const { return inputs.size(); }

const Matrix& Dataset::get_input(size_t index) const { return inputs[perm_idx[index]]; }

const size_t Dataset::get_output(size_t index) const { return outputs[perm_idx[index]]; }

void Dataset::shuffle() 
{ 
    std::shuffle(perm_idx.begin(), perm_idx.end(), std::mt19937{std::random_device{}()});
}