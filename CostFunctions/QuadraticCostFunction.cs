using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetworkMyself
{
    // Quadratic Cost Function
    // For a single training set:
    //   C = 1/2 * (desiredOutput - output)^2 = 1/2 * SUM((desiredOutput[j] - output[j])^2) over all vector elements j
    // For multiple training sets:
    //   C = 1/n * SUM(C_x) over all training sets x, where n = number of training sets
    //     = 1/2n * SUM((desiredOutput_x - output_x)^2)
    //     = 1/2n * SUM((desiredOutput_x[j] - output_x[j])^2) over all training sets x and vector elements j
    // Derivative w.r.t. output:
    //   C' = output - desiredOutput
    class QuadraticCostFunction : BaseCostFunction
    {
        public override string Name { get; set; }

        public QuadraticCostFunction() : base("QuadraticCostFunction") { }

        public QuadraticCostFunction(QuadraticCostFunction otherFunction) : base(otherFunction) { }

        // Computes the cost function based on the achieved output array and the desired output array
        public override float Compute(Vector<float> output, Vector<float> desiredOutput)
        {
            return (float)(0.5 * (desiredOutput - output).PointwisePower(2).Sum());
        }

        // Compute cost function for minibatch - each column of the matrices represents a separate training set
        public override float Compute(Matrix<float> output, Matrix<float> desiredOutput)
        {
            return (float)(1.0 / (2 * output.ColumnCount) * (desiredOutput - output).PointwisePower(2).ColumnSums().Sum());
        }

        // Computes the partial derivative of the cost function with respect to 'output' for minibatch
        public override Matrix<float> ComputeDerivate(Matrix<float> output, Matrix<float> desiredOutput)
        {
            return output - desiredOutput;
        }
    }
}
