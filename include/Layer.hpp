// layer.hpp

#pragma once
#include "Matrix.hpp"

enum class Activation {
    None,
    ReLU,
    Softmax
};

class Layer
{
    protected:
        Matrix A;
        Matrix b;
        Matrix W;
        Matrix Z;

        Matrix dA;
        Matrix db;
        Matrix dW;
        Matrix dZ;

        const Matrix& prev_A;
        Matrix* prev_dA;
        
    public:
        Layer(size_t input_size, size_t output_size, const Matrix& prev_A, Matrix* prev_dA);
        ~Layer() = default;

        // Getters
        const Matrix& getA() const;
        const Matrix& get_dA() const;
        Matrix& get_dA();
        const Matrix& get_dZ() const;

        // Setters
        void setA(const Matrix& g);
        void set_dA(const Matrix& g);
        void set_dZ(const Matrix& g);

        virtual void forward() = 0;
        virtual void backprop() = 0;

        void step(double learning_rate);
};

class HiddenLayer final : public Layer
{
    public:
        HiddenLayer(size_t input_size, size_t output_size, const Matrix& prev_A, Matrix* prev_dA);

        void forward() override;
        void backprop() override;
};

class OutputLayer : public Layer
{
    public:
        OutputLayer(size_t input_size, size_t output_size, const Matrix& prev_A, Matrix* prev_dA);

        void forward() override;
        void backprop() override;
};
