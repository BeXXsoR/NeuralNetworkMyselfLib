using MathNet.Numerics.LinearAlgebra;
using System;

namespace NeuralNetworkMyself
{
    class MyBipolarSigmoidFunction : BaseActivationFunction
    {
        public override string Name { get; set; }
        private float alpha = 1;

        public float Alpha { get; set; }

        public MyBipolarSigmoidFunction() : base("MyBipolarSigmoidFunction") { }

        public MyBipolarSigmoidFunction(float alpha) : base("MyBipolarSigmoidFunction")
        {
            this.alpha = alpha;
        }

        public MyBipolarSigmoidFunction(MyBipolarSigmoidFunction otherFunction) : base(otherFunction)
        {
            alpha = otherFunction.alpha;
        }

        public override float Function(float x)
        {
            return (float)((2 / (1 + Math.Exp(-alpha * x))) - 1);
        }

        public override Matrix<float> Function(Matrix<float> z)
        {
            // BipolarSigmoid = 2 / ( 1 + exp(-az)) - 1  =  2 / ( 1 + 1 / exp(az)) - 1  =  2 / ((exp(az) + 1) / exp(az)) - 1 = 2 * exp(az) / (exp(az) + 1) - 1
            //  = (2 * exp(az) - exp(az) - 1) / (exp(az) + 1)  =  (exp(az) - 1) / (exp(az) + 1)
            Matrix<float> ePowAlphaZ = EPowAlphaZ(z);
            return (ePowAlphaZ - 1).PointwiseDivide(ePowAlphaZ + 1);
        }

        public override void Function(Matrix<float> z, out Matrix<float> result)
        {
            Matrix<float> ePowAlphaZ = EPowAlphaZ(z);
            result = (ePowAlphaZ - 1).PointwiseDivide(ePowAlphaZ + 1);
        }

        public override float Derivative(float x)
        {
            float y = Function(x);
            return (alpha * (1 - y * y) / 2);
        }
        public override Matrix<float> Derivative(Matrix<float> z)
        {
            //bisigmoid' = a * (1 - bisig^2) / 2 = a * (1 - ((exp(az) - 1) / (exp(az) + 1))^2) / 2
            //  = a * (1 - (exp(az) - 1) ^ 2 / (exp(az) + 1) ^ 2) / 2
            //  = a * ((exp(az) + 1) ^ 2 - (exp(az) - 1) ^ 2) / (exp(az) + 1) ^ 2) / 2
            //  = a * (4 * exp(az)) / (exp(az) + 1) ^ 2) / 2
            //  = a * 2 * exp(az) / (exp(az) + 1) ^ 2)
            Matrix<float> ePowAlphaZ = EPowAlphaZ(z);
            return (2 * Alpha) * ePowAlphaZ.PointwiseDivide((ePowAlphaZ + 1).PointwisePower(2));
        }

        public override float Derivative2(float y)
        {
            return (alpha * (1 - y * y) / 2);
        }

        private Matrix<float> EPowAlphaZ(Matrix<float> z)
        {
            return Alpha == 1 ? z.PointwiseExp() : (Alpha * z).PointwiseExp();
        }
    }
}
