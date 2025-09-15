// dataset.cpp

#include "Dataset.hpp"

Dataset::Dataset(std::vector<Matrix> inputs, std::vector<size_t> outputs)
{
    if (inputs.size() != outputs.size())
    {
        throw std::invalid_argument("Error: Inputs and outputs must have the same size");
    }

    this->inputs = inputs;
    this->outputs = outputs;
}

const size_t Dataset::size() const { return inputs.size(); }

const Matrix& Dataset::get_input(size_t index) const { return inputs[index]; }

const size_t Dataset::get_output(size_t index) const { return outputs[index]; }

void Dataset::shuffle() 
{ 
    // TODO: Implement shuffling
}