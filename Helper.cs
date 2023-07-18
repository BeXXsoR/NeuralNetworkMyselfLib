using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;

namespace NeuralNetworkMyself
{
    public static class Helper
    {

        public static int[] indices = new int[0];

        public enum LearningRateAdjType
        {
            Absolute,
            Relative,
            Nothing
        }

        public enum ActivationFunctionType
        {
            Sigmoid,
            BipolarSigmoid,
            Identity,
            ReLu,
            Softmax
        }

        public enum CostFunctionType
        {
            Quadratic,
            CrossEntropy,
            LogLikelihood
        }

        public enum RegularizationType
        {
            NoRegularization,
            L2Regularization
        }

        public enum EarlyStoppingType
        {
            Off,
            Simple,
            Advanced
        }

        public enum MessageTrigger
        {
            NoNetwork,
            OutputHeader,
            UpdateResults,
            TrainingFinished,
            DisplayParam
        }

        // Matrix erzeugen, bei der jede Spalte aus demselben Vektor besteht
        public static Matrix<float> BuildMatrixOfColumnVector(Vector<float> vector, int numColumns)
        {
            //Idee: Inputvektor mal [1,1, ... ,1]-Vektor transponiert (numColumns lang) ergibt die gesuchte Matrix
            Vector<float> multiplier = Vector<float>.Build.Dense(numColumns, 1);
            return vector.ToColumnMatrix() * multiplier.ToRowMatrix();
        }

        //In-place Version der Methode
        public static void BuildMatrixOfColumnVector(Vector<float> vector, int numColumns, out Matrix<float> result)
        {
            //Idee: Inputvektor mal [1,1, ... ,1]-Vektor transponiert (numColumns lang) ergibt die gesuchte Matrix
            Vector<float> multiplier = Vector<float>.Build.Dense(numColumns, 1);
            result = vector.ToColumnMatrix() * multiplier.ToRowMatrix();
        }

        // Matrix erzeugen, bei der jede Zeile aus demselben Vektor besteht
        public static Matrix<float> BuildMatrixOfRowVector(Vector<float> vector, int numRows)
        {
            Vector<float> multiplier = Vector<float>.Build.Dense(numRows, 1);
            return multiplier.ToColumnMatrix() * vector.ToRowMatrix();
        }

        // Fisher Yates Algorithm zum Mischen der Spalten einer Matrix (in-place-suffling)
        public static void ShuffleColumnsOfMatricesUsingFisherYates(Matrix<float>[] matrices)
        {
            Random rnd = new Random();
            for (int i = matrices[0].ColumnCount - 1; i > 0; i--)
            {
                int swapWithPos = rnd.Next(i + 1);
                // Do the same swap in all matrices
                for (int j = 0; j < matrices.Length; j++)
                {
                    Vector<float> helper = matrices[j].Column(i);
                    matrices[j].SetColumn(i, matrices[j].Column(swapWithPos));
                    matrices[j].SetColumn(swapWithPos, helper);
                }
            }
        }

        // Performace-Test - Fisher Yates wie oben, aber wir mischen nur die Indizes und erzeugen daraus eine neue Matrix
        public static Matrix<float> ShuffleColumnsOfMatrixUsingFisherYatesNewMatrix(Matrix<float> matrix)
        {
            if (indices.Length != matrix.ColumnCount)
            {
                indices = new int[matrix.ColumnCount];
                for (int i = 0; i < indices.Length; i++)
                    indices[i] = i;
            }

            int[] newOrderedIndices = new int[indices.Length];
            indices.CopyTo(newOrderedIndices, 0);
            Random rnd = new Random();
            for (int i = newOrderedIndices.Length - 1; i > 0; i--)
            {
                int swapWithPos = rnd.Next(i + 1);
                int helper = newOrderedIndices[i];
                newOrderedIndices[i] = newOrderedIndices[swapWithPos];
                newOrderedIndices[swapWithPos] = helper;
            }

            List<Vector<float>> listOfNewColumns = new List<Vector<float>>();
            for (int i = 0; i < newOrderedIndices.Length; i++)
            {
                listOfNewColumns.Add(matrix.Column(newOrderedIndices[i]));
            }
            return Matrix<float>.Build.DenseOfColumnVectors(listOfNewColumns);
        }
    }
}
