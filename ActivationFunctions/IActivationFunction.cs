using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetworkMyself
{

    public interface IActivationFunction
    {
        float Function(float x);

        // Calculates function value for whole Matrix where each column represents the neuron vector of a distinct training set
        Matrix<float> Function(Matrix<float> z);

        //In-place version of above method
        void Function(Matrix<float> z, out Matrix<float> result);

        float Derivative(float x);

        // Calculates function derivative for whole Matrix where each column represents the neuron vector of a distinct training set
        Matrix<float> Derivative(Matrix<float> z);

        float Derivative2(float y);

        // Preferred way of weight initialization w.r.t. this activation function
        IContinuousDistribution WeightInitialization(int numInputNodesInPriorLayer = 0);
    }
}
