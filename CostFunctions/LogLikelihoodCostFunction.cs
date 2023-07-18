using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworkMyself
{
    class LogLikelihoodCostFunction : BaseCostFunction
    {
        // Log-Likelihood cost function
        // Assumption: desiredOutput is a vector with exactly one '1' and otherwise '0'
        // For a single training set:
        //   C = -ln(output[i]), where i is the index of the '1' in desiredOutput
        //     = -ln(SUM(output * desiredOutput) over all i), where '*' means the Hadamard Product (Pointwise Multiply)
        //        instead of SUM, MAX works as well
        // For multiple training sets:
        //   C = 1/n * SUM(C_x) over all training sets x, where n = number of training sets
        //     = -1/n * SUM(ln(SUM((output * desiredOutput)[ij] over all rows i) over all columns j)
        // Derivative at position ij:
        //   C' = 0 if i is the index of a '0' in desiredOutput for training set j, and
        //        -1/output[i] if i is the index of the '1' in desiredOutput for training set j
        public override string Name { get; set; }

        public LogLikelihoodCostFunction() : base("LogLikelihoodCostFunction") { }

        public override float Compute(Vector<float> output, Vector<float> desiredOutput)
        {
            for (int i = 0; i < desiredOutput.Count; i++)
            {
                if (desiredOutput[i] == 1)
                {
                    return (float)-Math.Log(output[i]);
                }
            }
            //// Ist das hier schneller?
            //return Math.Log(output.PointwiseMultiply(desiredOutput).Maximum());
                
            // Default-Returnwert: einfach den ersten nehmen
            return (float)Math.Log(output[0]); 
        }

        public override float Compute(Matrix<float> output, Matrix<float> desiredOutput)
        {
            return (float)((-1.0 / output.ColumnCount) * output.PointwiseMultiply(desiredOutput).ColumnSums().PointwiseLog().Sum());
        }

        public override Matrix<float> ComputeDerivate(Matrix<float> output, Matrix<float> desiredOutput)
        {
            return output.DivideByThis(-1.0f).PointwiseMultiply(desiredOutput);
        }
    }
}
