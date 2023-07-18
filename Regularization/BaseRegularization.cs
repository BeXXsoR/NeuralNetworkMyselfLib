using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetworkMyself
{
    public abstract class BaseRegularization : IRegularization
    {
        abstract public string Name { get; set;  }
        abstract public float Lambda {get; set; }

        public BaseRegularization(string name, float lambda)
        {
            Name = name;
            Lambda = lambda;
        }

        public BaseRegularization(BaseRegularization otherRegularization)
        {
            Name = otherRegularization.Name;
            Lambda = otherRegularization.Lambda;
        }

        abstract public float Compute(Matrix<float>[] weights, int numSets);

        abstract public Matrix<float> ComputeDerivative(Matrix<float> weights, int numSets);

        abstract public float ComputeWeightDecayFactor(int numTrainingSets);
    }
}
