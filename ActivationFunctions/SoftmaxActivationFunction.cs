using MathNet.Numerics.LinearAlgebra;
using System;

namespace NeuralNetworkMyself
{
    class SoftmaxActivationFunction : BaseActivationFunction
    {
        public SoftmaxActivationFunction() : base("SoftmaxActivationFunction") { }

        
        // softmax(z_i) = exp(z_i) / SUM(exp(z_k) over all neurons k in that layer)
        public override float Function(float x)
        {
            throw new ArgumentException("Can't use SoftmaxActivationFunction.Function on single value.");
        }

        public override Matrix<float> Function(Matrix<float> z)
        {
            // z_ij -> exp(z_ij) / SUM(exp(z_kj) over all k = 1 to numRows)
            return z.PointwiseExp().NormalizeColumns(1.0);
        }

        public override void Function(Matrix<float> z, out Matrix<float> result)
        {
            // z_ij -> exp(z_ij) / SUM(exp(z_kj) over all k = 1 to numRows)
            result = z.PointwiseExp();
            result = result.NormalizeColumns(1.0);
        }

        // softmax'(z_i) = exp'(z_i) * SUM(exp(z_k)) - exp(z_i) * SUM'(exp(z_k)) / SUM(exp(z_k))^2
        //  = exp(z_i) * SUM(exp(z_k)) - exp(z_i) * exp(z_i) / SUM(exp(z_k)^2
        //  = exp(z_i) * SUM(exp(z_k) over all k!=i) / SUM(exp(z_k)^2
        public override float Derivative(float x)
        {
            throw new ArgumentException("Can't use SoftmaxActivationFunction.Derivative on single value.");
        }

        public override Matrix<float> Derivative(Matrix<float> z)
        {
            //z_ij -> exp(z_ij) * (SUM(exp(z_kj) over all k) - exp(z_ij)) / (SUM(exp(z_kj) over all k)^2
            Matrix<float> ePowZ = z.PointwiseExp();
            Vector<float> ePowZColumnSumsVector = ePowZ.ColumnSums();
            Matrix<float> ePowZColumnSums = Helper.BuildMatrixOfRowVector(ePowZColumnSumsVector, z.RowCount);
            Matrix<float> ePowZColumnSumsSquared = ePowZColumnSums.PointwisePower(2);
            return ePowZ.PointwiseMultiply(ePowZColumnSums - z).PointwiseDivide(ePowZColumnSumsSquared);
        }

        public override float Derivative2(float y)
        {
            throw new ArgumentException("Can't use SoftmaxActivationFunction.Derivative2 on single value.");
        }
    }
}
