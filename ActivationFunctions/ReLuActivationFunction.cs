using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using System;

namespace NeuralNetworkMyself
{
    class ReLuActivationFunction : BaseActivationFunction
    {
        public override string Name { get; set; }

        public ReLuActivationFunction() : base("ReLuActivationFunction") { }

        public ReLuActivationFunction(ReLuActivationFunction otherFunction) : base(otherFunction) { }

        public override float Function(float x)
        {
            if (x >= 0)
                return x;
            else
                return 0;
        }

        public override Matrix<float> Function(Matrix<float> z)
        {
            return z.PointwiseMaximum(0);
        }

        public override void Function(Matrix<float> z, out Matrix<float> result)
        {
            result = z.PointwiseMaximum(0);
        }

        public override float Derivative(float x)
        {
            if (x >= 0)
                return 1;
            else
                return 0;
        }

        public override Matrix<float> Derivative(Matrix<float> z)
        {
            return z.PointwiseSign().PointwiseMaximum(0);
        }

        public override float Derivative2(float y)
        {
            if (y >= 0)
                return 1;
            else
                return 0;
        }

        public override IContinuousDistribution WeightInitialization(int numInputNodesInPriorLayer = 0)
        {
            // not sure if this makes sense, but at least all the values are positive and between 0 and 1
            return new ContinuousUniform(0, 0.1 / Math.Sqrt(numInputNodesInPriorLayer));

            // Use Kaiming algorithm
            //return new Normal(0, Math.Sqrt(2.0 / numInputNodesInPriorLayer));
        }

        public override IContinuousDistribution BiasesInitialization(int numInputNodesInPriorLayer = 0)
        {
            // not sure if this makes sense, but at least all the values are positive and between 0 and 1
            // Take the same distribution as in Weightinitialization. I want to have the Biases all positive to avoid negative z at the beginning, and I want them small enough to avoid z being too big.
            return new ContinuousUniform(0, 0.1 / Math.Sqrt(numInputNodesInPriorLayer));

            // Use Kaiming - initialize Biases to zero.
            //return new Normal(0, 0);
        }
    }
}
