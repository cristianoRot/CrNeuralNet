// dataset.hpp

#pragma once
#include "Matrix.hpp"
#include <vector>

class Dataset 
{
    private:   
        std::vector<Matrix> inputs;
        std::vector<size_t> outputs;

    public:
        Dataset(std::vector<Matrix> inputs, std::vector<size_t> outputs);

        const size_t size() const;

        const Matrix& get_input(size_t index) const;
        const size_t get_output(size_t index) const;

        void shuffle();
};
