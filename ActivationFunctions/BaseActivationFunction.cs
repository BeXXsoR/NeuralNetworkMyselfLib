using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetworkMyself
{
    abstract public class BaseActivationFunction : IActivationFunction
    {
        virtual public string Name { get; set; }

        public BaseActivationFunction(string name)
        {
            Name = name;
        }

        public BaseActivationFunction(BaseActivationFunction otherFunction)
        {
            Name = otherFunction.Name;
        }

        abstract public float Function(float x);

        abstract public Matrix<float> Function(Matrix<float> x);

        abstract public void Function(Matrix<float> z, out Matrix<float> result);

        abstract public float Derivative(float x);
        abstract public Matrix<float> Derivative(Matrix<float> x);

        abstract public float Derivative2(float y);

        virtual public IContinuousDistribution WeightInitialization(int numInputNodesInPriorLayer = 0)
        {
            return new Normal(0, 1);
        }

        virtual public IContinuousDistribution BiasesInitialization(int numInputNodesInPriorLayer = 0)
        {
            return new Normal(0, 1);
        }
    }
}
