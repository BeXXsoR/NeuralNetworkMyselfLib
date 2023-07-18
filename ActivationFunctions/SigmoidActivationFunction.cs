using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using System;

namespace NeuralNetworkMyself
{
    class SigmoidActivationFunction : BaseActivationFunction
    {
        public override string Name { get; set; }
        public float Alpha { get; }

        public SigmoidActivationFunction() : base("SigmoidActivationFunction")
        {
            Alpha = 1;        
        }

        public SigmoidActivationFunction(float alpha) : base("SigmoidActivationFunction")
        {
            Alpha = alpha;
        }

        public SigmoidActivationFunction(SigmoidActivationFunction otherFunction) : base(otherFunction)
        {
            Alpha = otherFunction.Alpha;
        }

        public override float Function(float x)
        {
            return (float)(1 / (1 + Math.Exp(-Alpha * x)));
        }

        public override Matrix<float> Function(Matrix<float> z)
        {
            // Sigmoid(z) = 1 / (1 + exp(-az)) = 1 / (1 + 1 / exp(az)) = 1 / ((exp(az) + 1) / exp(az)) = exp(az) / (exp(az) + 1)
            Matrix<float> ePowAlphaZ = EPowAlphaZ(z);
            return ePowAlphaZ.PointwiseDivide(ePowAlphaZ + 1);
        }

        public override void Function(Matrix<float> z, out Matrix<float> result)
        {
            Matrix<float> ePowAlphaZ = EPowAlphaZ(z);
            result = ePowAlphaZ.PointwiseDivide(ePowAlphaZ + 1);
        }

        public override float Derivative(float x)
        {
            float y = Function(x);
            return (Alpha * y * (1 - y));
        }

        public override Matrix<float> Derivative(Matrix<float> z)
        {
            // sigmoid' = a * sigmoid * (1 - sigmoid) = a * exp(az) / (exp(az) + 1) * (1 - exp(az) / (exp(az) + 1))
            //  = a * exp(az) / (exp(az) + 1) * (1 / (exp(az) + 1))  =  a * exp(az) / (exp(az) + 1) ^ 2
            Matrix<float> ePowAlphaZ = EPowAlphaZ(z);
            Matrix<float> DerWOAlphaMultiplication = ePowAlphaZ.PointwiseDivide((ePowAlphaZ + 1).PointwisePower(2));
            return Alpha == 1 ? DerWOAlphaMultiplication : Alpha * DerWOAlphaMultiplication;
        }

        public override float Derivative2(float y)
        {
            return (Alpha * y * (1 - y));
        }

        public override IContinuousDistribution WeightInitialization(int numInputNodesInPriorLayer)
        {
            return new Normal(0, 1.0 / Math.Sqrt(numInputNodesInPriorLayer));
        }

        private Matrix<float> EPowAlphaZ(Matrix<float> z)
        {
            return Alpha == 1 ? z.PointwiseExp() : (Alpha * z).PointwiseExp();
        }
    }
}
