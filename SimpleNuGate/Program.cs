using System;
using System.Collections.Generic;
using Encog.ML.Data.Basic;

namespace SimpleNuGate
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            EvaluateNuNetForORGate();
            Console.WriteLine("\n");
            EvaluateNuNetForANDGate();
            Console.ReadKey();
        }

        /// <summary>
        /// Construct a simple Neural Network and train with given datasets
        /// </summary>
        /// <returns></returns>
        private static NuNet GetNuNet(double [][] input, double [][] idealOutput)
        {
            var nuNet = new NuNet(input, idealOutput);
            // Train the neural network
            Console.Write("Training...");
            var trainingResult = nuNet.Train();
            Console.WriteLine($"\t\t[Iterations to train {trainingResult.Iterations}]\n");
            return nuNet;
        }

        /// <summary>
        /// Construct a simple Neural Network and train as a OR gate. Evaluate
        /// </summary>
        /// <returns></returns>
        private static void EvaluateNuNetForORGate()
        {
            var inputList = new[]
            {
                new[] {0.0, 0.0},
                new[] {1.0, 0.0},
                new[] {0.0, 1.0},
                new[] {1.0, 1.0}
            };

            // Expected output
            double [][] idealList =
            {
                new [] {0.0, 0},
                new [] {1.0, 0},
                new [] {1.0, 0},
                new [] {1.0, 0},
            };

            Console.WriteLine("----------------------------------------- OR Gate -----------------------------------------");
            using (var nuNet = GetNuNet(inputList, idealList))
            {
                Console.WriteLine("NuNet predictions for inputs :");
                var result = nuNet.Predict(new[] { 0.0, 0.0 });
                Console.WriteLine($"[0 | 0]: {Math.Round(result)} (Actual: {result})");

                result = nuNet.Predict(new[] { 0.0, 1.0 });
                Console.WriteLine($"[0 | 1]: {Math.Round(result)} (Actual: {result})");

                result = nuNet.Predict(new[] { 1.0, 0.0 });
                Console.WriteLine($"[1 | 0]: {Math.Round(result)} (Actual: {result})");

                result = nuNet.Predict(new[] { 1.0, 1.0 });
                Console.WriteLine($"[1 | 1]: {Math.Round(result)} (Actual: {result})");
            }
        }

        /// <summary>
        /// Construct a simple Neural Network and train as an AND gate
        /// </summary>
        /// <returns></returns>
        private static void EvaluateNuNetForANDGate()
        {
            var inputList = new[]
            {
                new[] {0.0, 0.0},
                new[] {1.0, 0.0},
                new[] {0.0, 1.0},
                new[] {1.0, 1.0}
            };

            // Expected output
            double[][] idealList =
            {
                new [] {0.0, 0},
                new [] {0.0, 0},
                new [] {0.0, 0},
                new [] {1.0, 0},
            };

            Console.WriteLine("----------------------------------------- AND Gate -----------------------------------------");
            using (var nuNet = GetNuNet(inputList, idealList))
            {
                Console.WriteLine("NuNet predictions for inputs :");
                var result = nuNet.Predict(new[] { 0.0, 0.0 });
                Console.WriteLine($"[0 | 0]: {Math.Round(result)} (Actual: {result})");

                result = nuNet.Predict(new[] { 0.0, 1.0 });
                Console.WriteLine($"[0 | 1]: {Math.Round(result)} (Actual: {result})");

                result = nuNet.Predict(new[] { 1.0, 0.0 });
                Console.WriteLine($"[1 | 0]: {Math.Round(result)} (Actual: {result})");

                result = nuNet.Predict(new[] { 1.0, 1.0 });
                Console.WriteLine($"[1 | 1]: {Math.Round(result)} (Actual: {result})");
            }
        }
    }
}