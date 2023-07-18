using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using static NeuralNetworkMyself.Helper;

namespace NeuralNetworkMyself
{
    // Klon eines Neural-Network-Objekts, um die relevanten Eigenschaften zu serialisieren, inkl. der Matrizen
    public class NNCloneForSerialization
    {
        public int NumInputNodes { get; set; }
        public int NumOutputNodes { get; set; }
        public int NumComputingLayers { get; set; }
        public int[] NumNodesPerComputingLayer { get; set; }
        public int SizeOfMiniBatch { get; set; }
        public float LearningRate { get; set; }
        public float Momentum { get; set; }
        public string[] ActivationFunctionNames { get; set; }
        public string CostFunctionName { get; set; }
        public string RegularizationName { get; set; }
        public float Lambda { get; set; }
        public bool IsActiveShuffling { get; set; }
        public EarlyStoppingType EarlyStoppingType { get; set; }
        public float LearningRateAdjustmentFactor { get; set; }
        public float NumMaxLearningRateAdjustments { get; set; }
        public DateTime AsOf { get; set; }
        public int BestSoFarEpoch { get; set; }
        public float BestSoFarAccuracyValidation { get; set; }
        public float[][][] Weights { get; set; }
        public float[][][] Velocities { get; set; }
        public float[][] Biases { get; set; }
        public float[][][] BestSoFarWeights { get; set; }
        public float[][][] BestSoFarVelocities { get; set; }
        public float[][] BestSoFarBiases { get; set; }
        public List<float> CostPerEpochTraining { get; set; }
        public List<float> CostPerEpochTesting { get; set; }
        public List<float> AccuracyPerEpochTraining { get; set; }
        public List<float> AccuracyPerEpochTesting { get; set; }
        public List<float> AccuracyPerEpochValidation { get; set; }


        //Constructor für Serialisierung (wird aufgerufen aus bestehender NeuralNetwork-Klasse, wenn NN gespeichert werden soll)
        public NNCloneForSerialization(NeuralNetwork network)
        {
            NumInputNodes = network.NumInputNodes;
            NumOutputNodes = network.NumOutputNodes;
            NumComputingLayers = network.NumComputingLayers;
            NumNodesPerComputingLayer = network.NumNodesPerComputingLayer;
            BestSoFarEpoch = network.BestSoFarEpoch;
            CostFunctionName = network.CostFunction is null ? "" : network.CostFunction.Name;
            ActivationFunctionNames = new string[NumComputingLayers];
            for (int i = 0; i < NumComputingLayers; i++)
                ActivationFunctionNames[i] = network.ActivationFunctions[i] is null ? "" : network.ActivationFunctions[i].Name;
            RegularizationName = network.Regularization is null ? "" : network.Regularization.Name;
            Lambda = network.Regularization is null ? 0 : network.Regularization.Lambda;
            IsActiveShuffling = network.IsActiveShuffling;
            SizeOfMiniBatch = network.SizeOfMiniBatch;
            LearningRate = network.LearningRate;
            Momentum = network.Momentum;
            CostPerEpochTraining = network.CostPerEpochTraining;
            CostPerEpochTesting = network.CostPerEpochTesting;
            AccuracyPerEpochTraining = network.AccuracyPerEpochTraining;
            AccuracyPerEpochTesting = network.AccuracyPerEpochTesting;
            AccuracyPerEpochValidation = network.AccuracyPerEpochValidation;
            BestSoFarAccuracyValidation = network.BestSoFarAccuracyValidation;
            EarlyStoppingType = network.EarlyStoppingType;
            LearningRateAdjustmentFactor = network.LearningRateAdjustmentFactor;
            NumMaxLearningRateAdjustments = network.NumMaxLearningRateAdjustments;
            AsOf = DateTime.Now;

            // Weights & Biases & Velocities
            Biases = new float[NumComputingLayers][];
            Weights = new float[NumComputingLayers][][];
            Velocities = new float[NumComputingLayers][][];
            BestSoFarBiases = new float[NumComputingLayers][];
            BestSoFarWeights = new float[NumComputingLayers][][];
            BestSoFarVelocities = new float[NumComputingLayers][][];
            for (int i = 0; i < NumComputingLayers; i++)
            {
                Biases[i] = network.Biases[i].ToArray();
                Weights[i] = network.Weights[i].ToColumnArrays();
                Velocities[i] = network.Velocities[i].ToColumnArrays();
                BestSoFarBiases[i] = network.BestSoFarBiases[i].ToArray();
                BestSoFarWeights[i] = network.BestSoFarWeights[i].ToColumnArrays();
                BestSoFarVelocities[i] = network.BestSoFarVelocities[i].ToColumnArrays();
            }
        }

        // Constructor for JSON Deserialization
        public NNCloneForSerialization() {}


        public void Serialize(string fileName)
        {
            AsOf = DateTime.Now;
            if (!(fileName is null))
            {
                File.WriteAllText(fileName, JsonSerializer.Serialize(this));
                return;
            }
            else
                return;
        }
    }
}
