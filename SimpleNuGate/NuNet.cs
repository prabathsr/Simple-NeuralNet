using System;
using System.Linq;
using Encog.Engine.Network.Activation;
using Encog.ML.Data.Basic;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using Encog.Neural.Networks.Training.Propagation.Back;

namespace SimpleNuGate
{
    internal class NuNet : IDisposable
    {
        private readonly bool _isDisposed = false;
        private readonly BasicNetwork _network;
        private readonly double[][] _idealOutput;
        private readonly double[][] _input;

        private BasicMLDataSet _trainingSet;

        public NuNet(double[][] input, double[][] idealOutput)
        {
            _input = input;
            _idealOutput = idealOutput;
            InitTrainingData();
            _network = CreateNetwork();
        }

        public void Dispose()
        {
            if (_isDisposed)
                return;

            _trainingSet.Close();
            _network.ClearContext();
        }

        private void InitTrainingData()
        {
            _trainingSet = new BasicMLDataSet(_input, _idealOutput);
        }

        /// <summary>
        ///     Create a very simple Neural Network with 3 Layers using ENCOG framework
        ///     * Layer 1: Input layer with 2 input neurones and 1 bias neurones
        ///     * Layer 2: Hidden layer with 2 neurones and 1 bias neurones and Sigmoid Activation function
        ///     * Layer 3: Output layer with 2 output neurones and Sigmoid Activation function
        /// </summary>
        /// <returns></returns>
        private BasicNetwork CreateNetwork()
        {
            var network = new BasicNetwork();
            network.AddLayer(new BasicLayer(null, true, 2));
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 2));
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), false, 2));
            network.Structure.FinalizeStructure();
            network.Reset();

            return network;
        }

        /// <summary>
        ///     Training neural network with datasets
        /// </summary>
        public ITrainingResult Train()
        {
            var train = new Backpropagation(_network, _trainingSet, 0.7, 0.8);

            var count = 0;

            // Train the neural network with input and ideal output data
            do
            {
                count++;
                train.Iteration();
            } while (train.Error > 0.001);

            return new NuTrainingResult
            {
                Iterations = count,
                Error = train.Error
            };
        }

        public double Predict(double[] input)
        {
            var output = new double[2];
            _network.Compute(input, output);
            return output.First();
        }
    }
}