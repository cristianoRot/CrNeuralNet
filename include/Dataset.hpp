// dataset.hpp

#pragma once
#include "Matrix.hpp"
#include <vector>

class Dataset 
{
    private:   
        std::vector<Matrix> inputs;
        std::vector<Matrix> outputs;

    public:
        Dataset(std::vector<Matrix> inputs, std::vector<Matrix> outputs);

        const size_t size() const;

        const Matrix& get_input(size_t index) const;
        const Matrix& get_output(size_t index) const;

        void shuffle();
};
