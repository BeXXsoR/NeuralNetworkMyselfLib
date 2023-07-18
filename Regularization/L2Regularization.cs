using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkMyself
{
    // Class for the L2 regularization
    public class L2Regularization : BaseRegularization
    {
        public override string Name { get; set; }
        public override float Lambda { get; set; }

        public L2Regularization(float lambda) : base("L2Regularization", lambda) { }

        public L2Regularization(BaseRegularization otherRegularization) : base(otherRegularization) { }

        public override float Compute(Matrix<float>[] weights, int numTrainingSets)
        {
            float sum = 0;
            for (int i = 0; i < weights.Length; i++)
            {
                sum += (float)Math.Pow(weights[i].FrobeniusNorm(), 2);
            }
            
            return sum * 0.5f * Lambda / numTrainingSets;
        }

        public override Matrix<float> ComputeDerivative(Matrix<float> weights, int numTrainingSets)
        {
            return (Lambda / numTrainingSets) * weights;
        }

        public override float ComputeWeightDecayFactor(int numTrainingSets)
        {
            return (Lambda / numTrainingSets);
        }
    }
}
