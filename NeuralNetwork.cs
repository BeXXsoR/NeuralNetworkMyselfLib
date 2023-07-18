using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using static NeuralNetworkMyself.Helper;


namespace NeuralNetworkMyself
{
    // This class uses the notation described here: http://neuralnetworksanddeeplearning.com/chap2.html
    public class NeuralNetwork
    {
        #region Class member
        public int NumInputNodes { get; set; }
        public int NumOutputNodes { get; set; }
        public int NumComputingLayers { get; set; }
        // Computing means hidden and output layer
        public int[] NumNodesPerComputingLayer { get; set; }
        public Matrix<float>[] Weights { get; set; }
        // The array describes the layers (hidden plus output). Per Layer i we have a weight matrix
        // with NumNodesPerLayer[i] rows and NumNodesPerLayer[i-1] columns
        public Matrix<float>[] Velocities { get; set; }
        // The array describes the layers (hidden plus output). Per Layer i we have a velocity matrix
        // with NumNodesPerLayer[i] rows and NumNodesPerLayer[i-1] columns (i.e. one velocity per weight)
        public Vector<float>[] Biases { get; set; }
        // The array describes the layers (hidden plus output). Per Layer i we have a bias vector with NumNodesPerLayer[i] entries
        public Matrix<float>[] OutputZ { get; set; }
        // The array describes the layers (hidden plus output). Per Layer i we have a matrix with NumNodesPerLayer[i] rows and
        // sizOfMiniBatch columns. Each column represents the Z-vector for the respective training example in the mini batch
        public Matrix<float>[] OutputAlpha { get; set; }
        // The array describes the layers (hidden plus output). Per Layer i we have a matrix with NumNodesPerLayer[i] rows and
        // sizOfMiniBatch columns. Each column represents the Alpha-vector for the respective training example in the mini batch
        public Matrix<float>[] Errors { get; set; }
        // The array describes the layers (hidden plus output). Per Layer i we have a matrix with NumNodesPerLayer[i] rows and
        // sizeOfMiniBatch columns. Each column represents the Error-vector for the respective training example in the mini batch
        public Matrix<float>[] WeightDeactivation { get; set; }
        // The array describes the layers (hidden plus output). Per Layer i, this matrix dimensions match the weight matrix dimensions.
        // This matrix is initially populated from the outside, and it acts as a pointwise multiplicator to the weight matrix after each backpropagation. 
        // By that one can simulate weights to be non-existing (by putting a '0' at the respective entries in this matrix and a '1' in all other entries)
        public Matrix<float>[] BestSoFarWeights { get; set; }
        public Matrix<float>[] BestSoFarVelocities { get; set; }
        public Vector<float>[] BestSoFarBiases { get; set; }
        public int BestSoFarEpoch { get; set; }
        // The best epoch is measured on the validation data
        public BaseCostFunction CostFunction { get; set; }
        public BaseActivationFunction[] ActivationFunctions { get; set; }
        public BaseRegularization Regularization { get; set; }
        public int SizeOfMiniBatch { get; set; }
        public float LearningRate { get; set; }
        public float Momentum { get; set; }
        public List<float> CostPerEpochTraining { get; set; }
        public List<float> CostPerEpochTesting { get; set; }
        public List<float> AccuracyPerEpochTraining { get; set; }
        public List<float> AccuracyPerEpochTesting { get; set; }
        public List<float> AccuracyPerEpochValidation { get; set; }
        public float BestSoFarAccuracyValidation { get; set; }
        public EarlyStoppingType EarlyStoppingType { get; set; }
        public float LearningRateAdjustmentFactor { get; set; }
        public float NumMaxLearningRateAdjustments { get; set; }
        public bool IsActiveShuffling { get; set; }
        Vector<float> InputTrainingMeans { get; set; }
        Vector<float> InputTrainingStdDev { get; set; }
        Matrix<float> InputTraining { get; set; }
        Matrix<float> OutputTraining { get; set; }
        Matrix<float> InputTesting { get; set; }
        Matrix<float> OutputTesting { get; set; }
        Matrix<float> InputValidation { get; set; }
        Matrix<float> OutputValidation { get; set; }
        private Stopwatch stopwatch { get; set; }
        // Helper-Matrizen, damit wir beim Berechnen des Outputs ohne Updaten nicht immer die Matrizen neue initialisieren müssen
        public Matrix<float>[] HelperOutputZ { get; set; }
        public Matrix<float>[] HelperOutputAlpha { get; set; }
        public int HelperLastInputSize { get; set; }
        #endregion

        #region Constructor & Initialization
        // Constructor
        public NeuralNetwork(int numInputNodes, int[] numNodesPerComputingLayer, int sizeOfMiniBatch, float learningRate, float momentum, EarlyStoppingType earlyStoppingType, 
            BaseCostFunction costFunction, BaseActivationFunction[] activationFunction, BaseRegularization regularization, float lambda = 0, bool isActiveShuffling = false, float learningRateAdjustmentFactor = 1, float numMaxLearningRateAdjustments = 0)
        {
            BasicInitializion(numInputNodes, numNodesPerComputingLayer, sizeOfMiniBatch, learningRate, momentum, earlyStoppingType, isActiveShuffling, learningRateAdjustmentFactor, numMaxLearningRateAdjustments);
            InitializeFunctions(costFunction, activationFunction, regularization, lambda);
            InitializeMatrices();
        }

        // Constructor mit Namen der Funktionen (statt der Funktionen selbst)
        public NeuralNetwork(int numInputNodes, int[] numNodesPerComputingLayer, int sizeOfMiniBatch, float learningRate, float momentum, EarlyStoppingType earlyStoppingType,
            string costFuncName, string[] activationFuncNames, string regulName, float lambda = 0, bool isActiveShuffling = false, float learningRateAdjustmentFactor = 1, float numMaxLearningRateAdjustments = 0)
        {
            BasicInitializion(numInputNodes, numNodesPerComputingLayer, sizeOfMiniBatch, learningRate, momentum, earlyStoppingType, isActiveShuffling, learningRateAdjustmentFactor, numMaxLearningRateAdjustments);
            InitializeFunctions(costFuncName, activationFuncNames, regulName, lambda);
            InitializeMatrices();
        }
        
        // Constructor mit Namen der Funktionen (statt der Funktionen selbst) - nur eine Activation Function
        public NeuralNetwork(int numInputNodes, int[] numNodesPerComputingLayer, int sizeOfMiniBatch, float learningRate, float momentum, EarlyStoppingType earlyStoppingType,
            string costFuncName, string activationFuncName, string regulName, float lambda = 0, bool isActiveShuffling = false, float learningRateAdjustmentFactor = 1, float numMaxLearningRateAdjustments = 0)
        {
            BasicInitializion(numInputNodes, numNodesPerComputingLayer, sizeOfMiniBatch, learningRate, momentum, earlyStoppingType, isActiveShuffling, learningRateAdjustmentFactor, numMaxLearningRateAdjustments);
            string[] actFuncNames = new string[NumComputingLayers];
            for (int i = 0; i < actFuncNames.Length; i++)
                actFuncNames[i] = activationFuncName;
            InitializeFunctions(costFuncName, actFuncNames, regulName, lambda);
            InitializeMatrices();
        }

        //Constructor mittels Deserialisierung aus JSON-Datei
        public NeuralNetwork(string fileName)
        {
            DeserializeNN(fileName);
        }

        // Einfache Parameter initialisieren
        public void BasicInitializion(int numInputNodes, int[] numNodesPerComputingLayer, int sizeOfMiniBatch, float learningRate, float momentum, EarlyStoppingType earlyStoppingType,
            bool isActiveShuffling, float learningRateAdjustmentFactor, float numMaxLearningRateAdjustments)
        {
            NumInputNodes = numInputNodes;
            NumNodesPerComputingLayer = numNodesPerComputingLayer;
            NumComputingLayers = NumNodesPerComputingLayer.Length;
            NumOutputNodes = NumNodesPerComputingLayer[NumComputingLayers - 1];
            SizeOfMiniBatch = sizeOfMiniBatch;
            LearningRate = learningRate;
            Momentum = momentum;
            BestSoFarEpoch = 0;
            IsActiveShuffling = isActiveShuffling;
            CostPerEpochTraining = new List<float>();
            CostPerEpochTesting = new List<float>();
            AccuracyPerEpochTraining = new List<float>();
            AccuracyPerEpochTesting = new List<float>();
            AccuracyPerEpochValidation = new List<float>();
            BestSoFarAccuracyValidation = 0.0f;
            EarlyStoppingType = earlyStoppingType;
            LearningRateAdjustmentFactor = learningRateAdjustmentFactor;
            NumMaxLearningRateAdjustments = numMaxLearningRateAdjustments;
            stopwatch = new Stopwatch();
            HelperLastInputSize = 0;
        }

        // Functions initialisieren mit schon erstellten Funktionen
        public void InitializeFunctions(BaseCostFunction costFunction, BaseActivationFunction[] activationFunction, BaseRegularization regularization, float lambda = 0)
        {
            CostFunction = costFunction;
            ActivationFunctions = activationFunction;
            Regularization = regularization;
            SetLambdaInRegularization(lambda);
        }
        
        // Functions initialisieren mit den Funktionsnamen
        public void InitializeFunctions(string costFuncName, string[] activationFuncName, string regulName, float lambda)
        {
            SetCostFunction(costFuncName);
            SetActivationFunction(activationFuncName);
            SetRegularization(regulName, lambda);
        }

        // Matrizen initialisieren
        public void InitializeMatrices()
        {
            // Weights and Biases initialisieren
            Weights = new Matrix<float>[NumComputingLayers];
            Velocities = new Matrix<float>[NumComputingLayers];
            Biases = new Vector<float>[NumComputingLayers];
            OutputZ = new Matrix<float>[NumComputingLayers];
            OutputAlpha = new Matrix<float>[NumComputingLayers];
            Errors = new Matrix<float>[NumComputingLayers];
            BestSoFarWeights = new Matrix<float>[NumComputingLayers];
            BestSoFarVelocities = new Matrix<float>[NumComputingLayers];
            BestSoFarBiases = new Vector<float>[NumComputingLayers];
            WeightDeactivation = new Matrix<float>[NumComputingLayers];
            HelperOutputZ = new Matrix<float>[NumComputingLayers];
            HelperOutputAlpha = new Matrix<float>[NumComputingLayers];
            for (int i = 0; i < NumComputingLayers; i++)
            {
                // Weights: use random numbers from the distribution that is specific to the activation function (if available)
                // Velocities: initialize with zeroes
                // Biases:  use random numbers from a Gaussian distribution with mean 0 and standard deviation 1
                int numInputNodesInPriorLayer = i == 0 ? NumInputNodes : NumNodesPerComputingLayer[i - 1];
                Weights[i] = Matrix<float>.Build.Random(NumNodesPerComputingLayer[i], numInputNodesInPriorLayer, ActivationFunctions[i] is null ? new Normal(0, 1) : ActivationFunctions[i].WeightInitialization(numInputNodesInPriorLayer));
                Velocities[i] = Matrix<float>.Build.Dense(NumNodesPerComputingLayer[i], numInputNodesInPriorLayer);
                Biases[i] = Vector<float>.Build.Random(NumNodesPerComputingLayer[i], ActivationFunctions[i] is null ? new Normal(0, 1) : ActivationFunctions[i].BiasesInitialization(numInputNodesInPriorLayer));
                OutputZ[i] = Matrix<float>.Build.Dense(NumNodesPerComputingLayer[i], SizeOfMiniBatch);
                OutputAlpha[i] = Matrix<float>.Build.Dense(NumNodesPerComputingLayer[i], SizeOfMiniBatch);
                Errors[i] = Matrix<float>.Build.Dense(NumNodesPerComputingLayer[i], SizeOfMiniBatch);
                BestSoFarWeights[i] = Matrix<float>.Build.Dense(NumNodesPerComputingLayer[i], numInputNodesInPriorLayer);
                BestSoFarVelocities[i] = Matrix<float>.Build.Dense(NumNodesPerComputingLayer[i], numInputNodesInPriorLayer);
                BestSoFarBiases[i] = Vector<float>.Build.Dense(NumNodesPerComputingLayer[i]);
                WeightDeactivation[i] = null;
            }
        }

        public void SetCostFunction(string functionName)
        {
            switch (functionName)
            {
                case "Quadratic":
                case "QuadraticCostFunction":
                    CostFunction = new QuadraticCostFunction();
                    return;
                case "Cross Entropy":
                case "CrossEntropy":
                case "CrossEntropyCostFunction":
                    CostFunction = new CrossEntropyCostFunction();
                    return;
                case "Log Likelihood":
                case "Log-Likelihood":
                case "LogLikelihood":
                case "LogLikelihoodCostFunction":
                    CostFunction = new LogLikelihoodCostFunction();
                    return;
                default:
                    return;
            }
        }

        // Set activation function for given layer
        public void SetActivationFunction(int layerId, string functionName)
        {
            switch (functionName)
            {
                case "ReLu":
                case "ReLuActivationFunction":
                    ActivationFunctions[layerId] = new ReLuActivationFunction();
                    return;
                case "Identity":
                case "IdentityActivationFunction":
                    ActivationFunctions[layerId] = new IdentityActivationFunction();
                    return;
                case "BipolarSigmoid":
                case "MyBipolarSigmoidFunction":
                    ActivationFunctions[layerId] = new MyBipolarSigmoidFunction();
                    return;
                case "Sigmoid":
                case "SigmoidActivationFunction":
                    ActivationFunctions[layerId] = new SigmoidActivationFunction();
                    return;
                case "Softmax":
                case "SoftmaxActivationFunction":
                    ActivationFunctions[layerId] = new SoftmaxActivationFunction();
                    return;
                default:
                    return;
            }
        }

        // Set same activation function for all layers
        public void SetActivationFunction(string functionName)
        {
            ActivationFunctions = new BaseActivationFunction[NumComputingLayers]; 
            for (int i = 0; i < NumComputingLayers; i++)
                SetActivationFunction(i, functionName);
        }

        // Set activation functions based on function names
        public void SetActivationFunction(string[] functionNames)
        {
            ActivationFunctions = new BaseActivationFunction[NumComputingLayers];
            for ( int i = 0; i < Math.Min(NumComputingLayers, functionNames.Length); i++)
            {
                SetActivationFunction(i, functionNames[i]);
            }
        }

        public void SetRegularization(string regularizationName, float lambda)
        {
            switch (regularizationName)
            {
                case "L2 Regularization":
                case "L2Regularization":
                    Regularization = new L2Regularization(lambda);
                    return;
                case "No Regularization":
                case "NoRegularization":
                    Regularization = null;
                    return;
                default:
                    return;
            }
        }

        public void SetLambdaInRegularization(float lambda)
        {
            if (!(Regularization is null))
                Regularization.Lambda = lambda;
        }

        public void PopulateMatrices(Matrix<float> inputTraining, Matrix<float> outputTraining, Matrix<float> inputTesting, Matrix<float> outputTesting, Matrix<float> inputValidation, Matrix<float> outputValidation, Matrix<float>[] weightActivePassiveMgmt)
        {
            InputTraining = Matrix<float>.Build.DenseOfMatrix(inputTraining);
            InputTesting = Matrix<float>.Build.DenseOfMatrix(inputTesting);
            InputValidation = Matrix<float>.Build.DenseOfMatrix(inputValidation);
            OutputTraining = Matrix<float>.Build.DenseOfMatrix(outputTraining); 
            OutputTesting = Matrix<float>.Build.DenseOfMatrix(outputTesting);
            OutputValidation = Matrix<float>.Build.DenseOfMatrix(outputValidation);
            // Initialize WeightsActivePassiveMgmt and multiply it with the weight matrix
            if (!(weightActivePassiveMgmt is null))
            {
                for (int i = 0; i < weightActivePassiveMgmt.Length; i++)
                    if (!(weightActivePassiveMgmt[i] is null))
                    {
                        WeightDeactivation[i] = weightActivePassiveMgmt[i].Clone();
                        DeactivateSpecficiWeights(i);                        
                    }
            }            
        }

        public void CalculateInputTrainingMeansAndStdDev(Matrix<float> inputTraining)
        {
            //  mean = SUM(x_i over all training sets i) / n
            //  std = Sqrt(SUM((x_i - mean)^2 over all training sets i) / (n-1))
            InputTrainingMeans = inputTraining.RowSums().Divide(inputTraining.ColumnCount);
            InputTrainingStdDev = inputTraining.Subtract(BuildMatrixOfColumnVector(InputTrainingMeans, inputTraining.ColumnCount)).PointwisePower(2).RowSums().PointwiseSqrt().Divide(inputTraining.ColumnCount - 1);
        }

        public Matrix<float> StandardizeInputDataToStandardNormalDistribution(Matrix<float> input)
        {
            // See here: http://www.faqs.org/faqs/ai-faq/neural-nets/part2/ (search for "Formulas are as follows")
            //   mean = SUM(x_i over all training sets i) / #trainingSets
            //   std = Sqrt(SUM((x_i - mean)^2 over all training sets i) / (n-1))
            //   s_i = (x_i - mean) / std (NOte: Do PointwiseMaximum on std first to avoid dividing by zero)
            float minStdDev = 0.000000001f;
            if (InputTrainingMeans is null || InputTrainingStdDev is null)
                return input;
            else
                return input.Subtract(BuildMatrixOfColumnVector(InputTrainingMeans, input.ColumnCount)).PointwiseDivide(BuildMatrixOfColumnVector(InputTrainingStdDev.PointwiseMaximum(minStdDev), input.ColumnCount));
        }

        // BestSoFarMatrizen aktivieren - bisherige Matrizen werden nicht gespeichert!
        public void ActivateBestSoFarMatrices()
        {
            for (int i = 0; i < Weights.Length; i++)
            {
                BestSoFarWeights[i].CopyTo(Weights[i]);
                BestSoFarVelocities[i].CopyTo(Velocities[i]);
                BestSoFarBiases[i].CopyTo(Biases[i]);
            }
        }
        #endregion

        #region Core training methods
        //Train network using the populated training and testing matrices
        public void TrainNN(bool isConsoleApp, int numEpochs, bool computeCostPerEpoch = false, bool computeAccuracyPerEpoch = false,
            bool displayAccuracyTraining = true, bool displayAccuracyTesting = true, bool displayAccuracyValidation = true)
        {
            // First, compute Accuracy at beginning
            if (computeAccuracyPerEpoch)
            {
                ComputeClassificationAccuracy(displayAccuracyTraining, displayAccuracyTesting, displayAccuracyValidation);
                if (isConsoleApp)
                {
                    PrintoutValidationAccuracy(displayAccuracyTraining, displayAccuracyTesting, displayAccuracyValidation);
                }
            }
            int numMiniBatchRunsPerEpoch = InputTraining.ColumnCount / SizeOfMiniBatch;
            int numTargetEpochs = numEpochs;
            int numLearningRateAdjustments = 0;
            stopwatch.Restart();
            for (int curEpoch = 1; curEpoch <= numTargetEpochs; curEpoch++)
            {
                if (IsActiveShuffling)
                {
                    // Shuffle the training data sets
                    ShuffleColumnsOfMatricesUsingFisherYates(new Matrix<float>[2] { InputTraining, OutputTraining });
                }
                // Start mini batch
                int curStartColumn = 0;
                for (int curRun = 0; curRun < numMiniBatchRunsPerEpoch; curRun++)
                {
                    Matrix<float> curInput = InputTraining.SubMatrix(0, InputTraining.RowCount, curStartColumn, Math.Min(SizeOfMiniBatch, InputTraining.ColumnCount - curStartColumn));
                    Matrix<float> curOutput = OutputTraining.SubMatrix(0, OutputTraining.RowCount, curStartColumn, Math.Min(SizeOfMiniBatch, OutputTraining.ColumnCount - curStartColumn));
                    RunMiniBatch(curInput, curOutput);
                    curStartColumn += SizeOfMiniBatch;
                }
                // Compute cost if desired
                if (computeCostPerEpoch)
                {
                    float curCostTraining = ComputeCost(ComputeOutputWithoutUpdating(InputTraining), OutputTraining);
                    float curCostTesting = ComputeCost(ComputeOutputWithoutUpdating(InputTesting), OutputTesting);
                    if (float.IsNaN(curCostTraining) || float.IsNaN(curCostTesting))
                        // Cost is exploded, probably LearningRate too high
                        Debugger.Break();
                    CostPerEpochTraining.Add(curCostTraining);
                    CostPerEpochTesting.Add(curCostTesting);
                }
                // Compute accuracy if desired and check for improvements
                if (computeAccuracyPerEpoch)
                {
                    ComputeClassificationAccuracy(displayAccuracyTraining, displayAccuracyTesting, displayAccuracyValidation);
                    float curAccuracyValidation = AccuracyPerEpochValidation.Last();
                    if (curAccuracyValidation > BestSoFarAccuracyValidation)
                    {
                        BestSoFarAccuracyValidation = curAccuracyValidation;
                        BestSoFarEpoch = curEpoch;
                        for (int i = 0; i < NumComputingLayers; i++)
                        {
                            BestSoFarWeights[i] = Weights[i].Clone();
                            BestSoFarVelocities[i] = Velocities[i].Clone();
                            BestSoFarBiases[i] = Biases[i].Clone();
                        }
                        // Adjust target epoch number in early stopping mode
                        if (EarlyStoppingType != EarlyStoppingType.Off)
                            numTargetEpochs = curEpoch + numEpochs;
                    }
                    // Print intermediate results if app is console app
                    if (isConsoleApp)
                    {
                        PrintoutValidationAccuracy(displayAccuracyTraining, displayAccuracyTesting, displayAccuracyValidation);
                    }
                }
                // Check if Learning Rate needs to be adjusted in advanced early stopping mode
                if (EarlyStoppingType == EarlyStoppingType.Advanced && curEpoch == numTargetEpochs)
                {
                    LearningRate /= LearningRateAdjustmentFactor;
                    numLearningRateAdjustments++;
                    if (numLearningRateAdjustments <= NumMaxLearningRateAdjustments)
                    {
                        numTargetEpochs = curEpoch + numEpochs;
                    }
                }
            }
            stopwatch.Stop();
        }

        public void ContinueTraining(int numEpochs, bool isConsoleApp)
        {
            TrainNN(isConsoleApp, numEpochs, false, true);
        }

        public void RunMiniBatch(Matrix<float> input, Matrix<float> desiredOutput)
        {
            ComputeOutputWithUpdating(input);
            ComputeAndBackpropagateError(desiredOutput);
            UpdateWeightsAndBiases(input);
        }
        #endregion
       
        #region Backpropagation
        // Backpropagation - Error berechnen
        public void ComputeAndBackpropagateError(Matrix<float> desiredOutput)
        {
            int lastLayerId = NumComputingLayers - 1;
            // Zuerst Output Layer
            //   Im Fall von ActivationFunction = Sigmoid und CostFunction = CrossEntropy kürzt sich einiges weg, daher kann der Fehler vereinfacht berechnet werden
            if (ActivationFunctions[lastLayerId] is SigmoidActivationFunction && CostFunction is CrossEntropyCostFunction)
            {
                //ComputeActivationFunctionDerivative(OutputZ.Last()) ist hier:  (alpha * y * (1 - y)) mit y = sigmoid(z) = output;
                //CostFunction.ComputeDerivate(OutputAlpha.Last(), desiredOutput) ist hier:  (output - desiredOutput).PointwiseDivide(output.PointwiseMultiply(output.SubtractFrom(1)));
                //PointwiseMultiplay ergibt: alpha * (output - desiredOutput)
                Errors[lastLayerId] = (ActivationFunctions[lastLayerId] as SigmoidActivationFunction).Alpha * (OutputAlpha.Last() - desiredOutput);
            }
            //   Im Fall von ActivationFunction = Softmax und CostFunction = LogLikelihood kürzt sich ebenfalls einiges weg
            else if (ActivationFunctions[lastLayerId] is SoftmaxActivationFunction && CostFunction is LogLikelihoodCostFunction)
            {
                //ComputeActivationFunctionDerivative(OutputZ.Last()) ist hier:  exp(z_i) * SUM(exp(z_k)) - exp(z_i) * exp(z_i) / SUM(exp(z_k)^2
                //CostFunction.ComputeDerivate(OutputAlpha.Last(), desiredOutput) ist hier:  output.DivideByThis(-1.0).PointwiseMultiply(desiredOutput)
                //PointwiseMultiplay ergibt: output - desiredOutput
                Errors[lastLayerId] = OutputAlpha.Last() - desiredOutput;
            }
            else
            {
                Matrix<float> lastLayerActivationFunctionPrime = ComputeActivationFunctionDerivative(OutputZ.Last(), NumComputingLayers - 1);
                Matrix<float> lastLayerError = CostFunction.ComputeDerivate(OutputAlpha.Last(), desiredOutput);
                Errors[lastLayerId] = lastLayerError.PointwiseMultiply(lastLayerActivationFunctionPrime);
            }
            // Dann rückwärts alle weiteren Layer
            for (int curLayer = Errors.Length - 2; curLayer >= 0; curLayer--)
            {
                Matrix<float> curLayerActivationFunctionPrime = ComputeActivationFunctionDerivative(OutputZ[curLayer], curLayer);
                Matrix<float> curLayerError = Weights[curLayer + 1].TransposeThisAndMultiply(Errors[curLayer + 1]);
                Errors[curLayer] = curLayerError.PointwiseMultiply(curLayerActivationFunctionPrime);
            }
        }

        // Weights und Biases aktualisieren
        // Formel: siehe http://neuralnetworksanddeeplearning.com/chap2.html, just above "The Code for Backpropagation", step 3. Gradient descent
        public void UpdateWeightsAndBiases(Matrix<float> inputTrainingSet)
        {
            // Schleife über alle Layer
            for (int curLayer = NumComputingLayers - 1; curLayer >= 0; curLayer--)
            {
                // Gradient berechnen
                Matrix<float> gradientWeights = Errors[curLayer].TransposeAndMultiply(curLayer == 0 ? inputTrainingSet : OutputAlpha[curLayer - 1]);
                Vector<float> gradientBiases = Errors[curLayer].RowSums();
                // Velocities, Weights und Biases aktualisieren
                // Weights Update in einzelne if-Zweige aufbrechen, um Performance zu optimieren (so wenig Matrix-Operationen wie möglich)
                if (Momentum == 0 && Regularization is null)
                {
                    Weights[curLayer] -= (LearningRate / SizeOfMiniBatch) * gradientWeights;
                }
                else if (Momentum == 0 && !(Regularization is null)) 
                {
                    Weights[curLayer] = (float)(1.0 - LearningRate * Regularization.ComputeWeightDecayFactor(InputTraining.ColumnCount)) * Weights[curLayer] - (LearningRate / SizeOfMiniBatch) * gradientWeights;
                }
                else if (Momentum != 0)
                {
                    Velocities[curLayer] = Momentum * Velocities[curLayer] - (LearningRate / SizeOfMiniBatch) * gradientWeights; 
                    if (!(Regularization is null))
                        Velocities[curLayer] -= (LearningRate * Regularization.ComputeWeightDecayFactor(InputTraining.ColumnCount)) * Weights[curLayer];
                    Weights[curLayer] += Velocities[curLayer];
                }
                // Mulitply Weights with WeightActivePassiveMgmt
                DeactivateSpecficiWeights(curLayer);
                Biases[curLayer] -= (LearningRate / SizeOfMiniBatch) * gradientBiases;
            }
        }

        // Deactive certain weights by pointwise mulitplying the weight matrix with the weight deactivation matrix
        private void DeactivateSpecficiWeights(int layerId)
        {
            if (!(WeightDeactivation[layerId] is null))
                Weights[layerId].PointwiseMultiply(WeightDeactivation[layerId], Weights[layerId]);
        }
        #endregion
        
        #region Compute output
        // Output des Neuronalen Netzes berechnen für MiniBatch. Jede Spalte der Inputmatrix stellt einen Inputvector dar
        public Matrix<float> ComputeOutput(Matrix<float> input, Matrix<float>[] targetOutputZ, Matrix<float>[] targetOutputAlpha)
        {
            targetOutputZ[0] = Weights[0] * input;
            targetOutputZ[0] += BuildMatrixOfColumnVector(Biases[0], input.ColumnCount);
            targetOutputAlpha[0] = ComputeActivationFunction(targetOutputZ[0], 0);
            for (int i = 1; i < Weights.Length; i++)
            {
                targetOutputZ[i] = Weights[i] * targetOutputAlpha[i - 1];
                targetOutputZ[i] += BuildMatrixOfColumnVector(Biases[i], targetOutputAlpha[i].ColumnCount);
                targetOutputAlpha[i] = ComputeActivationFunction(targetOutputZ[i], i);
            }
            return targetOutputAlpha.Last();
        }

        // Output berechnen, ohne OutputZ und OutputAlpha zu aktualisieren (daher Nutzung von HelperOutputZ und HelperOutputAlpha)
        public Matrix<float> ComputeOutputWithoutUpdating(Matrix<float> input)
        {
            if (HelperLastInputSize != input.ColumnCount)
            {
                for (int i = 0; i < NumComputingLayers; i++)
                {
                    HelperOutputZ[i] = Matrix<float>.Build.Dense(NumNodesPerComputingLayer[i], input.ColumnCount);
                    HelperOutputAlpha[i] = Matrix<float>.Build.Dense(NumNodesPerComputingLayer[i], input.ColumnCount);
                }
                HelperLastInputSize = input.ColumnCount;
            }
            return ComputeOutput(input, HelperOutputZ, HelperOutputAlpha);
        }

        public Matrix<float> ComputeOutputWithUpdating(Matrix<float> input)
        {
            return ComputeOutput(input, OutputZ, OutputAlpha);
        }
        #endregion

        #region Compute Cost
        public float ComputeCost(Matrix<float> curOutput, Matrix<float> desiredOutput)
        {
            if (Regularization is null)
                return CostFunction.Compute(curOutput, desiredOutput);
            else
                return CostFunction.Compute(curOutput, desiredOutput) + Regularization.Compute(Weights, desiredOutput.ColumnCount);
        }

        public float ComputeCost(Vector<float> curOutput, Vector<float> desiredOutput)
        {
            if (Regularization is null)
                return CostFunction.Compute(curOutput, desiredOutput);
            else
                return CostFunction.Compute(curOutput, desiredOutput) + Regularization.Compute(Weights, OutputTraining.ColumnCount);
        }

        public void ComputeClassificationAccuracy(bool onTrainingSet, bool onTestingSet, bool onValidationSet)
        {
            if (onTrainingSet)
            {
                ComputeClassificationAccuracy(InputTraining, OutputTraining, AccuracyPerEpochTraining);
            }
            if (onTestingSet)
            {
                ComputeClassificationAccuracy(InputTesting, OutputTesting, AccuracyPerEpochTesting);
            }
            if (onValidationSet)
            {
                ComputeClassificationAccuracy(InputValidation, OutputValidation, AccuracyPerEpochValidation);
            }
        }

        public void ComputeClassificationAccuracy(Matrix<float> input, Matrix<float> expectedOutput, List<float> listToAddAccuracy)
        {
            Matrix<float> curOutput = ComputeOutputWithoutUpdating(input);
            int numCorrectIdentifications = 0;
            for (int j = 0; j < input.ColumnCount; j++)
            {
                if (curOutput.Column(j).MaximumIndex() == expectedOutput.Column(j).MaximumIndex())
                {
                    numCorrectIdentifications++;
                }
            }
            listToAddAccuracy.Add(numCorrectIdentifications / (float)input.ColumnCount);
        }
        #endregion

        #region Compute Activation
        // Activation Function auf ganze Matrix anwenden
        public Matrix<float> ComputeActivationFunction(Matrix<float> input, int layerId)
        {
            Matrix<float> output = ActivationFunctions[layerId].Function(input);
            return output;
        }

        //In-place Version der Methode
        public void ComputeActivationFunction(Matrix<float> input, int layerId, out Matrix<float> result)
        {
            ActivationFunctions[layerId].Function(input, out result);
        }

        // Activation Function Derivative auf ganze Matrix anwenden
        public Matrix<float> ComputeActivationFunctionDerivative(Matrix<float> input, int layerId)
        {
            //Matrix<float> output = Matrix<float>.Build.Dense(input.RowCount, input.ColumnCount,
            //    (i, j) => ActivationFunction.Derivative(input.At(i, j)));
            Matrix<float> output = ActivationFunctions[layerId].Derivative(input);

            return output;
        }
        #endregion

        #region Helper
 
        
        public void WriteResultsTestFunction()
        {
            List<float> resultList = new List<float>();
            Matrix<float>[] infoMatrices;
            Vector<float>[] infoVectors;
            for (int curCase = 0; curCase < 4; curCase++)
            {
                infoMatrices = null;
                infoVectors = null;
                switch (curCase)
                {
                    case 0:
                        infoMatrices = Errors;
                        break;
                    case 1:
                        infoMatrices = Weights;
                        break;
                    case 2:
                        infoVectors = Biases;
                        break;
                    case 3:
                        infoMatrices = OutputZ;
                        break;
                    default:
                        infoMatrices = null;
                        break;
                }
                if (infoMatrices is null)
                    for (int curLayer = 0; curLayer < infoVectors.Length; curLayer++)
                        resultList.AddRange(infoVectors[curLayer].ToArray());
                else
                    for (int curLayer = 0; curLayer < infoMatrices.Length; curLayer++)
                        resultList.AddRange(infoMatrices[curLayer].ToColumnMajorArray());
            }
            string resultString = resultList.ToString();
            //Debugger.Break();
        }

        #endregion

        #region Load, save and printouts
        // Serialize Neural Network
        public void SerializeNN(string fileName)
        {
            NNCloneForSerialization NNClone = new NNCloneForSerialization(this);
            NNClone.Serialize(fileName);
        }

        public void DeserializeNN(string fileName)
        {
            if (!(fileName is null))
            {
                NNCloneForSerialization NNClone = JsonSerializer.Deserialize<NNCloneForSerialization>(File.ReadAllText(fileName));
                NumInputNodes = NNClone.NumInputNodes;
                NumNodesPerComputingLayer = NNClone.NumNodesPerComputingLayer;
                NumComputingLayers = NNClone.NumNodesPerComputingLayer.Length;
                NumOutputNodes = NNClone.NumNodesPerComputingLayer[NumComputingLayers - 1];
                SizeOfMiniBatch = NNClone.SizeOfMiniBatch;
                LearningRate = NNClone.LearningRate;
                Momentum = NNClone.Momentum;
                IsActiveShuffling = NNClone.IsActiveShuffling;
                CostPerEpochTraining = NNClone.CostPerEpochTraining;
                CostPerEpochTesting = NNClone.CostPerEpochTesting;
                AccuracyPerEpochTraining = NNClone.AccuracyPerEpochTraining;
                AccuracyPerEpochTesting = NNClone.AccuracyPerEpochTesting;
                AccuracyPerEpochValidation = NNClone.AccuracyPerEpochValidation;
                BestSoFarEpoch = NNClone.BestSoFarEpoch;
                BestSoFarAccuracyValidation = NNClone.BestSoFarAccuracyValidation;
                SetCostFunction(NNClone.CostFunctionName);
                SetActivationFunction(NNClone.ActivationFunctionNames);
                SetRegularization(NNClone.RegularizationName, NNClone.Lambda);
                EarlyStoppingType = NNClone.EarlyStoppingType;
                LearningRateAdjustmentFactor = NNClone.LearningRateAdjustmentFactor;
                NumMaxLearningRateAdjustments = NNClone.NumMaxLearningRateAdjustments;

                // Weights & Biases
                Weights = new Matrix<float>[NumComputingLayers];
                Velocities = new Matrix<float>[NumComputingLayers];
                Biases = new Vector<float>[NumComputingLayers];
                BestSoFarWeights = new Matrix<float>[NumComputingLayers];
                BestSoFarVelocities = new Matrix<float>[NumComputingLayers];
                BestSoFarBiases = new Vector<float>[NumComputingLayers];
                OutputZ = new Matrix<float>[NumComputingLayers];
                OutputAlpha = new Matrix<float>[NumComputingLayers];
                Errors = new Matrix<float>[NumComputingLayers];
                HelperOutputZ = new Matrix<float>[NumComputingLayers];
                HelperOutputAlpha = new Matrix<float>[NumComputingLayers];
                HelperLastInputSize = 0;
                for (int i = 0; i < NumComputingLayers; i++)
                {
                    Biases[i] = Vector<float>.Build.DenseOfArray(NNClone.Biases[i]);
                    Weights[i] = Matrix<float>.Build.DenseOfColumnArrays(NNClone.Weights[i]);
                    Velocities[i] = Matrix<float>.Build.DenseOfColumnArrays(NNClone.Velocities[i]);
                    BestSoFarBiases[i] = Vector<float>.Build.DenseOfArray(NNClone.BestSoFarBiases[i]);
                    BestSoFarWeights[i] = Matrix<float>.Build.DenseOfColumnArrays(NNClone.BestSoFarWeights[i]);
                    BestSoFarVelocities[i] = Matrix<float>.Build.DenseOfColumnArrays(NNClone.BestSoFarVelocities[i]);
                    OutputZ[i] = Matrix<float>.Build.Dense(NumNodesPerComputingLayer[i], SizeOfMiniBatch);
                    OutputAlpha[i] = Matrix<float>.Build.Dense(NumNodesPerComputingLayer[i], SizeOfMiniBatch);
                    Errors[i] = Matrix<float>.Build.Dense(NumNodesPerComputingLayer[i], SizeOfMiniBatch);
                }
            }
        }

        public void PrintoutValidationAccuracy(bool displayAccuracyTraining, bool displayAccuracyTesting, bool displayAccuracyValidation)
        {
            string text = (AccuracyPerEpochValidation.Count - 1).ToString() + ";" + LearningRate;
            if (displayAccuracyTraining)
                text += ";" + AccuracyPerEpochTraining.Last();
            if (displayAccuracyTesting)
                text += ";" + AccuracyPerEpochTesting.Last();
            if (displayAccuracyValidation)
                text += ";" + AccuracyPerEpochValidation.Last();
            text += ";" + stopwatch.Elapsed.ToString();
            Console.WriteLine(text);
        }
        #endregion

        #region Outcommented stuff
        // Load Training Data from DB
        //public void loadNNTrainingDataFromDB()
        //{
        //    DatabaseCommunicator comm = new DatabaseCommunicator(Globals.AccessDBConnect4);
        //    List<string[]> dbContent = comm.ReadDataFromDB(Globals.AccessTableNameIntPos, null, "");
        //    // Einträge vorbereiten
        //    List<float[]> inputTraining = new List<float[]>();
        //    List<float[]> inputTesting = new List<float[]>();
        //    List<float[]> inputValidation = new List<float[]>();
        //    List<float[]> outputTraining = new List<float[]>();
        //    List<float[]> outputTesting = new List<float[]>();
        //    List<float[]> outputValidation = new List<float[]>();
        //    for (int i = 0; i < dbContent.Count; i++)
        //    {
        //        float[] inputs = Globals.GetInputsFromString(dbContent[i][0]);
        //        float[] stateForNN = Globals.ConvertInputsToStateTwoNeuronsPerSpot(inputs);
        //        int output = Convert.ToInt32(dbContent[i][1]);
        //        float[] outputForNN = Globals.ConvertDBOutputToNNOutputTwoNeuronsPerSpot(output);
        //        Globals.NNDatasetCategory datasetCategory = Globals.ComputeDatasetCategory(stateForNN);
        //        switch (datasetCategory)
        //        {
        //            case Globals.NNDatasetCategory.Training:
        //                inputTraining.Add(stateForNN);
        //                outputTraining.Add(outputForNN);
        //                break;
        //            case Globals.NNDatasetCategory.Testing:
        //                inputTesting.Add(stateForNN);
        //                outputTesting.Add(outputForNN);
        //                break;
        //            case Globals.NNDatasetCategory.Validation:
        //                inputValidation.Add(stateForNN);
        //                outputValidation.Add(outputForNN);
        //                break;
        //            default:
        //                Debugger.Break();
        //                break;
        //        }
        //    }
        //    // Matrizen für NN aufbauen
        //    latestInputTraining = Matrix<float>.Build.DenseOfColumnArrays(inputTraining);
        //    latestOutputTraining = Matrix<float>.Build.DenseOfColumnArrays(outputTraining);
        //    latestInputTesting = Matrix<float>.Build.DenseOfColumnArrays(inputTesting);
        //    latestOutputTesting = Matrix<float>.Build.DenseOfColumnArrays(outputTesting);
        //    latestInputValidation = Matrix<float>.Build.DenseOfColumnArrays(inputValidation);
        //    latestOutputValidation = Matrix<float>.Build.DenseOfColumnArrays(outputValidation);
        //}


        //// Save Neural Network to DB
        //public bool SaveNNParamToDB(string newFileName)
        //{
        //    try
        //    {
        //        if (Globals.CopyFile(Globals.AccessDBNNRepTemplate, newFileName))
        //        {
        //            DatabaseCommunicator dbComm = new DatabaseCommunicator(newFileName);
        //            // Tabellenspalten in Setup-Tabelle anpassen
        //            dbComm.AddComputingLayerColumnsInSetupTable(NumComputingLayers);
        //            // String für Inhalt der Setup-Tabelle aufbauen
        //            string[][] setupData = new string[1][];
        //            setupData[0] = new string[Globals.AccessDBNNRepNumOfStandardCols + NumComputingLayers * 2];
        //            setupData[0][0] = "NNConnect4" + DateTime.Now;          // Name
        //            setupData[0][1] = "";                                   // Description
        //            setupData[0][3] = NumInputNodes.ToString();             // NumInputNodes
        //            setupData[0][4] = NumOutputNodes.ToString();            // NumOutputNodes
        //            setupData[0][5] = NumComputingLayers.ToString();        // NumComputingLayers
        //            setupData[0][6] = LearningRate.ToString();              // LearningRate
        //            setupData[0][7] = SizeOfMiniBatch.ToString();           // SizeOfMiniBatch
        //            setupData[0][8] = ActivationFunction.GetType().Name;    // ActivationFunction
        //            setupData[0][9] = CostFunction.GetType().Name;          // CostFunction
        //            for (int i = 0; i < NumComputingLayers - 1; i++)        // TableNameWeightsCLx + TableNameBiasesCLx
        //            {
        //                setupData[0][10 + 2 * i] = "WeightsCL" + (i + 1);
        //                setupData[0][10 + 2 * i + 1] = "BiasesCL" + (i + 1);
        //            }
        //            dbComm.WriteDataToDB(Globals.AccessTableNameSetup, setupData);
        //        }
        //        else
        //            return false;
        //    }
        //    catch (Exception ex)
        //    {
        //        MessageBox.Show(ex.Message, "Fehler!", MessageBoxButtons.OK, MessageBoxIcon.Error);
        //        return false;
        //    }
        //    return true;
        //}
        #endregion
    }
}
