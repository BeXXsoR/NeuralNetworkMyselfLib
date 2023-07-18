using MathNet.Numerics.LinearAlgebra;
using System;

namespace NeuralNetworkMyself
{
    // Cross-Entropy Cost Function
    // For a single training set:
    //   C = - SUM[desiredOutput[j] * ln(output[j]) + (1 - desiredOutput[j]) * ln(1 - output[j])] over all vector elements j
    // For multiple training sets:
    //   C = 1/n * SUM(C_x) over all training sets x, where n = number of training sets
    //     = -1/n * SUM[desiredOutput_x * ln(output_x) + (1 - desiredOutput_x) * ln(1 - output_x)]
    //     = -1/n * SUM[desiredOutput_x[j] * ln(output_x[j]) + (1 - desiredOutput_x[j]) * ln(1 - output_x[j])] over all training sets x and vector elements j
    // Derivative w.r.t. output:
    //   C' = -1/n * SUM[desiredOutput_x * 1/output_x + (1 - desiredOutput_x) * 1/(1 - output_x) * (-1)]
    //      = -1/n * SUM[desiredOutput_x / output_x - (1 - desiredOutput_x) / (1 - ouput_x)]
    //      = -1/n * SUM[(desiredOutput_x  * (1 - output_x) - (1 - desiredOutput_x) * output_x) / (output_x * (1 - output_x))]
    //      = -1/n * SUM[(desiredOutput_x - output_x) / (output_x * (1 - output_x)]
    //      = 1/n * SUM[(output_x - desiredOutput_x) / (output_x * (1 - output_x)]
    //  If desiredOutput uses the sigmoid function, than using chain rule we would mulitply the whole term with sigmoid_prime. As sigmoid_prime = sigmoid * (1 - sigmoid),
    //  the denominator cancels out, and we get C' = -1/n * SUM(desiredOutput_x - output_x) (times x_j)
    class CrossEntropyCostFunction : BaseCostFunction
    {
        public override string Name { get; set; }

        public CrossEntropyCostFunction() : base("CrossEntropyCostFunction") { }

        public CrossEntropyCostFunction(QuadraticCostFunction otherFunction) : base(otherFunction) { }

        // Computes the cost function based on the achieved output array and the desired output array
        public override float Compute(Vector<float> output, Vector<float> desiredOutput)
        {
            return (float)(-1.0 * (desiredOutput.PointwiseMultiply(output.PointwiseLog()) + (desiredOutput.SubtractFrom(1).PointwiseMultiply(output.SubtractFrom(1).PointwiseLog()))).Sum());
        }

        // Compute cost function for minibatch - each column of the matrices represents a separate training set
        public override float Compute(Matrix<float> output, Matrix<float> desiredOutput)
        {
            return (float)(-1.0 / output.ColumnCount * (desiredOutput.PointwiseMultiply(output.PointwiseLog()) + (desiredOutput.SubtractFrom(1).PointwiseMultiply(output.SubtractFrom(1).PointwiseLog()))).ColumnSums().Sum());
        }

        // Computes the partial derivative of the cost function with respect to 'output' for minibatch
        public override Matrix<float> ComputeDerivate(Matrix<float> output, Matrix<float> desiredOutput)
        {
            // Frage: Wo ist 1/n hin? Brauchen wir das hier nicht mehr -> mit Algortihmus abgleichen (ist bei Quadratic Cost auch so) -> Antwort: Backprop nimmt die partiellen Ableitungen an jedem Punkt und 
            // bildet erst am Ende den Durchschnitt. M.a.W.: Das 1/n wird später im Algorithmus angewandt
            return (output - desiredOutput).PointwiseDivide(output.PointwiseMultiply(output.SubtractFrom(1)));
        }
    }
}
