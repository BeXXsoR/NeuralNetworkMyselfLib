using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using static NeuralNetworkMyself.Helper;

namespace NeuralNetworkMyself
{
    abstract class NNWrapper
    {
        public NeuralNetwork Network { get; set; }
        public Matrix<float> InputTraining { get; set; }
        public Matrix<float> OutputTraining { get; set; }
        public Matrix<float> InputTesting { get; set; }
        public Matrix<float> OutputTesting { get; set; }
        public Matrix<float> InputValidation { get; set; }
        public Matrix<float> OutputValidation { get; set; }
        public Matrix<float>[] WeightDeactivation { get; set; }
        public Stopwatch Stopwatch { get; set; }
        public bool IsCreatedNetwork { get; set; }
        protected List<string[]> DBContent {get; set;}

        public NNWrapper()
        {
            Network = null;
            InputTraining = null;
            OutputTraining = null;
            InputTesting = null;
            OutputTesting = null;
            InputValidation = null;
            OutputValidation = null;
            WeightDeactivation = null;
            Stopwatch = new Stopwatch();
            IsCreatedNetwork = false;
            DBContent = new List<string[]>();
        }

        public void StartTraining(int numEpochs)
        {
            try
            {
                if (Network is null)
                    ShowMessage(MessageTrigger.NoNetwork);
                else
                {
                    LoadDataIfNeeded();
                    LoadWeightDeactivationMatrix();
                    Stopwatch.Start();
                    FeedMatricesToNetwork();
                    ShowMessage(MessageTrigger.OutputHeader);
                    ContinueTraining(numEpochs);
                    Stopwatch.Stop();
                    ShowMessage(MessageTrigger.TrainingFinished);
                }
            }
            catch (Exception ex)
            {
                ShowErrorMessage(ex);
            }
        }

        public void LoadDataIfNeeded()
        {
            if (InputTraining is null || OutputTraining is null || InputTesting is null || OutputTesting is null || InputValidation is null || OutputValidation is null)
            {
                LoadAndNormalizeData();
            }
        }

        // Load data into class member matrices, incl. normalization into respective value range. Has to be overwritten in subclasses
        abstract public bool LoadAndNormalizeData(int maxRecords = 500000, bool keepDBContent = false);

        abstract public bool LoadWeightDeactivationMatrix();

        // Feed the current matrices to the network
        public void FeedMatricesToNetwork()
        {
            Network.PopulateMatrices(InputTraining, OutputTraining, InputTesting, OutputTesting, InputValidation, OutputValidation, WeightDeactivation);
            //Release the storage in this class
            InputTraining = null;
            OutputTraining = null;
            InputTesting = null;
            OutputTesting = null;
            InputValidation = null;
            OutputValidation = null;
            WeightDeactivation = null;
        }

        // Create the network with given parameters
        public NeuralNetwork CreateNetwork(int numInputNodes, int[] numNodesPerComputingLayer, int sizeOfMiniBatch, float learningRate, float momentum, EarlyStoppingType earlyStoppingType,
            string costFuncName, string[] actFuncNames, string regulName, float lambda = 0, bool isActiveShuffling = false, float learningRateAdjustmentFactor = 1, float numMaxLearningRateAdjustments = 0)
        {
            Network = new NeuralNetwork(numInputNodes, numNodesPerComputingLayer, sizeOfMiniBatch, learningRate, momentum, earlyStoppingType, costFuncName, actFuncNames, regulName, lambda, isActiveShuffling, learningRateAdjustmentFactor, numMaxLearningRateAdjustments);
            IsCreatedNetwork = true;
            return Network;
        }

        // Create the network from json file
        public NeuralNetwork CreateNetwork(string filename)
        {
            Network = new NeuralNetwork(filename);
            IsCreatedNetwork = true;
            return Network;
        }

        //Continue learning (or start it if it's the first time)
        public void ContinueTraining(int numEpochs)
        {
            try
            {
                Network.ContinueTraining(numEpochs, IsConsoleApp());
            }
            catch (Exception ex)
            {
                ShowErrorMessage(ex);
            }
        }

        // Update the results
        public void UpdateTrainingResults()
        {
            ShowMessage(MessageTrigger.UpdateResults);
        }

        // Save network to json file
        public void SaveNN(string filename)
        {
            Network.SerializeNN(filename);
        }

        // Use Swarm Intelligence
        abstract public void UseSwarmInTelligence();

        // Check if network is used in console app. Has to be overwritten in subclasses
        abstract public bool IsConsoleApp();
        
        // Display a message to the user. Has to be overwritten in subclasses.
        abstract public void ShowMessage(MessageTrigger trigger);

        abstract public void ShowErrorMessage(Exception ex);

        // Method for test stuff
        virtual public void TestStuff() { }
    }
}
