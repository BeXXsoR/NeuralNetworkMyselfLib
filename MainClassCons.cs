using System;
using static NeuralNetworkMyself.Helper;

namespace NeuralNetworkMyself
{
    // Call this class for a standard console app implementation of the neural network, with <T> being the subclass of NNWrapper that was implemented for the specific usecase
    static class MainClassCons
    {
        static public void ConsMain<T>(int numInputNodes = 784, int numComputingLayers = 2, int[] numNodesPerComputingLayer = null, int sizeOfMiniBatch = 10, int numEpochs = 10, EarlyStoppingType earlyStoppingType = EarlyStoppingType.Advanced,
            float learningRate = 0.1f, float momentum = 0.0f, string costFuncName = "CrossEntropy", string[] actFuncNames = null, string regulName = "NoRegularization", float lambda = 0.0f, bool isActiveShuffling = false,
            float learningRateAdjustmentFactor = 2, float numMaxLearningRateAdjustments = 5)
            where T : NNWrapper, new()
        {
            NNWrapper mainClass = new T();
            if (numNodesPerComputingLayer is null)
                numNodesPerComputingLayer = new int[2] { 30, 10 };
            if (actFuncNames is null)
                actFuncNames = new string[2] { "ReLu", "Sigmoid" };
            DisplayHelpMessage();
            DisplayParamNoNetwork();
            bool isEnd = false;
            char space = ' ';
            Console.WriteLine("\nCommand?");
            string command = Console.ReadLine();
            while (!isEnd)
            {
                string[] comArgs = command.Split(space);
                for (int i = 0; i < comArgs.Length; i++)
                {
                    string curArg = comArgs[i].ToLower();
                    switch (curArg)
                    {
                        case "createfromparam":
                            mainClass.CreateNetwork(numInputNodes, numNodesPerComputingLayer, sizeOfMiniBatch, learningRate, momentum, earlyStoppingType, costFuncName, actFuncNames, regulName, lambda, isActiveShuffling, learningRateAdjustmentFactor, numMaxLearningRateAdjustments);
                            break;
                        case "createfromfile":
                            mainClass.CreateNetwork(comArgs[i + 1]);
                            break;
                        case "start":
                            Console.WriteLine("Start training for " + numEpochs + " epochs.");
                            mainClass.StartTraining(numEpochs);
                            //Console.WriteLine("Time elapsed: " + mainClass.Stopwatch.Elapsed.ToString());
                            break;
                        case "continue":
                            Console.WriteLine("Continue training for " + numEpochs + " epochs.");
                            mainClass.ContinueTraining(numEpochs);
                            //Console.WriteLine("Time elapsed: " + mainClass.Stopwatch.Elapsed.ToString());
                            break;
                        case "end":
                            return;
                        case "display":
                            if (mainClass.IsCreatedNetwork)
                                DisplayParamNetwork();
                            else
                                DisplayParamNoNetwork();
                            break;
                        case "save":
                            mainClass.SaveNN(comArgs[i + 1]);
                            break;
                        case "test":
                            mainClass.TestStuff();
                            break;
                        case "swarm":
                            mainClass.UseSwarmInTelligence();
                            break;
                        case "-h":
                            DisplayHelpMessage();
                            break;
                        case "-numin":
                            numInputNodes = Convert.ToInt32(comArgs[i + 1]);
                            break;
                        case "-numcl":
                            numComputingLayers = Convert.ToInt32(comArgs[i + 1]);
                            if (numNodesPerComputingLayer.Length != numComputingLayers)
                            {
                                Array.Resize(ref numNodesPerComputingLayer, numComputingLayers);
                                Array.Resize(ref actFuncNames, numComputingLayers);
                            }
                            break;
                        case "-numncl":
                            for (int j = 0; j < numComputingLayers; j++)
                            {
                                numNodesPerComputingLayer[j] = Convert.ToInt32(comArgs[i + 1]);
                                i++;
                            }
                            break;
                        case "-mb":
                            sizeOfMiniBatch = Convert.ToInt32(comArgs[i + 1]);
                            if (mainClass.IsCreatedNetwork)
                                mainClass.Network.SizeOfMiniBatch = Convert.ToInt32(comArgs[i + 1]);
                            break;
                        case "-nume":
                            numEpochs = Convert.ToInt32(comArgs[i + 1]);
                            break;
                        case "-stop":
                            earlyStoppingType = (EarlyStoppingType)Enum.Parse(typeof(EarlyStoppingType), comArgs[i + 1]); 
                            if (mainClass.IsCreatedNetwork)
                                mainClass.Network.EarlyStoppingType = (EarlyStoppingType)Enum.Parse(typeof(EarlyStoppingType), comArgs[i + 1]);
                            break;
                        case "-lr":
                            learningRate = Convert.ToSingle(comArgs[i + 1]); 
                            if (mainClass.IsCreatedNetwork)
                                mainClass.Network.LearningRate = Convert.ToSingle(comArgs[i + 1]);
                            break;
                        case "-mm":
                            momentum = Convert.ToSingle(comArgs[i + 1]); 
                            if (mainClass.IsCreatedNetwork)
                                mainClass.Network.Momentum = Convert.ToSingle(comArgs[i + 1]);
                            break;
                        case "-cost":
                            costFuncName = comArgs[i + 1];
                            break;
                        case "-act":
                            for (int j = 0; j < numComputingLayers; j++)
                            {
                                actFuncNames[j] = comArgs[i + 1];
                                i++;
                            }
                            break;
                        case "-reg":
                            regulName = comArgs[i + 1];
                            break;
                        case "-lm":
                            lambda = Convert.ToSingle(comArgs[i + 1]);
                            break;
                        case "-sh":
                            isActiveShuffling = Convert.ToBoolean(comArgs[i + 1]);
                            if (mainClass.IsCreatedNetwork)
                                mainClass.Network.IsActiveShuffling = Convert.ToBoolean(comArgs[i + 1]);
                            break;
                        case "-lraf":
                            learningRateAdjustmentFactor = Convert.ToSingle(comArgs[i + 1]); 
                            if (mainClass.IsCreatedNetwork)
                                mainClass.Network.LearningRateAdjustmentFactor = Convert.ToSingle(comArgs[i + 1]);
                            break;
                        case "-lra":
                            numMaxLearningRateAdjustments = Convert.ToSingle(comArgs[i + 1]); 
                            if (mainClass.IsCreatedNetwork)
                                mainClass.Network.NumMaxLearningRateAdjustments = Convert.ToSingle(comArgs[i + 1]);
                            break;
                        default:
                            break;
                    }
                }
                Console.WriteLine("\nCommand?");
                command = Console.ReadLine();
            }

            void DisplayParamNoNetwork()
            {
                Console.WriteLine("\tnumInputNodes: " + numInputNodes.ToString());
                Console.WriteLine("\tnumComputingLayers: " + numComputingLayers.ToString());
                string numNodesPerCompLayerString = "";
                for (int i = 0; i < numNodesPerComputingLayer.Length; i++)
                    numNodesPerCompLayerString += numNodesPerComputingLayer[i].ToString() + " ";
                Console.WriteLine("\tnumNodesPerComputingLayer: " + numNodesPerCompLayerString);
                Console.WriteLine("\tsizeOfMiniBatch: " + sizeOfMiniBatch.ToString());
                Console.WriteLine("\tnumEpochs: " + numEpochs.ToString());
                Console.WriteLine("\tlearningRate: " + learningRate.ToString());
                Console.WriteLine("\tmomentum: " + momentum.ToString());
                Console.WriteLine("\tcostFuncName: " + costFuncName);
                string activationFunctions = "";
                for (int i = 0; i < numNodesPerComputingLayer.Length; i++)
                    activationFunctions += actFuncNames[i] + " ";
                Console.WriteLine("\tactFuncName: " + activationFunctions);
                Console.WriteLine("\tregulName: " + regulName);
                Console.WriteLine("\tlambda: " + lambda.ToString());
                Console.WriteLine("\tisActiveShuffling: " + isActiveShuffling.ToString());
                Console.WriteLine("\tearlyStoppingType: " + earlyStoppingType.ToString());
                Console.WriteLine("\tlearningRateAdjustmentFactor: " + learningRateAdjustmentFactor.ToString());
                Console.WriteLine("\tnumMaxLearningRateAdjustments: " + numMaxLearningRateAdjustments.ToString());
            }

            void DisplayParamNetwork()
            {
                Console.WriteLine("\tnumEpochs: " + numEpochs.ToString());
                mainClass.ShowMessage(MessageTrigger.DisplayParam);
            }
        }

        static void DisplayHelpMessage()
        {
            Console.WriteLine("\tArgument list:");
            Console.WriteLine("\t\"start\": \tStarte Training");
            Console.WriteLine("\t\"continue\": \tTraining fortsetzen");
            Console.WriteLine("\t\"end\": \t\tBeenden");
            Console.WriteLine("\t\"display\": \tAktuelle Parameter anzeigen");
            Console.WriteLine("\t\"swarm\": \tSchwarmintelligenz starten");
            Console.WriteLine("\t\"-h\": \t\tDiese Nachricht anzeigen");
            Console.WriteLine("\t\"-numin\": \tnumInputNodes");
            Console.WriteLine("\t\"-numcl\": \tnumComputingLayers");
            Console.WriteLine("\t\"-numncl\": \tnumNodesPerComputingLayer");
            Console.WriteLine("\t\"-mb\": \t\tsizeOfMiniBatch");
            Console.WriteLine("\t\"-nume\": \tnumEpochs");
            Console.WriteLine("\t\"-lr\": \t\tlearningRate");
            Console.WriteLine("\t\"-mm\": \t\tmomentum");
            Console.WriteLine("\t\"-cost\": \tcostFuncName");
            Console.WriteLine("\t\"-act\": \tactFuncName");
            Console.WriteLine("\t\"-reg\": \tregulName");
            Console.WriteLine("\t\"-lm\": \t\tlambda");
            Console.WriteLine("\t\"-sh\": \t\tisActiveShuffling");
            Console.WriteLine("\t\"-stop\": \tearlyStoppingType");
            Console.WriteLine("\t\"-lraf\": \tlearningRateAdjustmentFactor");
            Console.WriteLine("\t\"-lra\": \tnumMaxLearningRateAdjustments");
        }
    }
}
