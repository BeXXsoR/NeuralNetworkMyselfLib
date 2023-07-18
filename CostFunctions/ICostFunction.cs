using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetworkMyself
{
    public interface ICostFunction
    {
        // Computes the cost function based on the achieved output array and the desired output array
        float Compute(Vector<float> output, Vector<float> desiredOutput);

        // Compute cost function for minibatch
        float Compute(Matrix<float> output, Matrix<float> desiredOutput);

        // Computes the partial derivative of the cost function with respect to 'output' for minibatch
        Matrix<float> ComputeDerivate(Matrix<float> output, Matrix<float> desiredOutput);

    }
}
