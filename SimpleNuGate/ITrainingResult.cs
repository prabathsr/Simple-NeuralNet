namespace SimpleNuGate
{
    internal interface ITrainingResult
    {
        int Iterations { get; set; }
        double Error { get; set; }
    }
}