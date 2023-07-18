using MathNet.Numerics.LinearAlgebra;
using System;

namespace NeuralNetworkMyself
{
    [Serializable]
    class IdentityActivationFunction : BaseActivationFunction
    {
        public override string Name { get; set; }

        public IdentityActivationFunction() : base("IdentityActivationFunction") { }

        public IdentityActivationFunction(IdentityActivationFunction otherFunction) : base(otherFunction) { }

        public override float Function(float x)
        {
            return x;
        }
        
        public override Matrix<float> Function(Matrix<float> z)
        {
            return z;
        }

        public override void Function(Matrix<float> z, out Matrix<float> result)
        {
            result = z;
        }

        public override float Derivative(float x)
        {
            return 1.0f;
        }

        public override Matrix<float> Derivative(Matrix<float> z)
        {
            return Matrix<float>.Build.Dense(z.RowCount, z.ColumnCount, 1.0f);
        }

        public override float Derivative2(float y)
        {
            return 1.0f;
        }

    }
}
