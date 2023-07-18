using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetworkMyself
{
    public interface IRegularization
    {

        // Computes the regularization term based on the weight matrix
        float Compute(Matrix<float>[] weights, int numSets);

        // Computes the weight decay of the regularization based on the weight matrix 
        Matrix<float> ComputeDerivative(Matrix<float> weights, int numSets);

        // Computes the weight decay factor
        float ComputeWeightDecayFactor(int numTrainingSets);

    }
}
