using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetworkMyself
{
    abstract public class BaseCostFunction : ICostFunction
    {
        abstract public string Name { get; set; }

        public BaseCostFunction(string name)
        {
            Name = name;
        }

        public BaseCostFunction(BaseCostFunction otherFunction)
        {
            Name = otherFunction.Name;
        }
        abstract public float Compute(Vector<float> output, Vector<float> desiredOutput);

        abstract public float Compute(Matrix<float> output, Matrix<float> desiredOutput);

        abstract public Matrix<float> ComputeDerivate(Matrix<float> output, Matrix<float> desiredOutput);
    }
}
