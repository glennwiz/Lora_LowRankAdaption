using System;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;

class Program
{
    static void Main(string[] args)
    {
        bool runWithDebug = AskForDebugMode();

        if (runWithDebug)
        {
            RunWithDebug();
        }
        else
        {
            RunWithoutDebug();
        }
    }

    static bool AskForDebugMode()
    {
        Console.WriteLine("Do you want to run in debug mode? (y/n)");
        string input = Console.ReadLine();

        return (input.ToLower() == "y");
    }

    static void RunWithDebug()
    {
        Console.WriteLine("Initializing a 100x100 matrix with random values between 0 and 1...");
        var rng = new System.Random();
        var matrix = Matrix<double>.Build.Dense(100, 100, (i, j) => rng.NextDouble());
        Console.WriteLine("Original Matrix (100x100 with random values):");
        Console.WriteLine(matrix.ToString());

        Console.WriteLine("Computing the Singular Value Decomposition (SVD) of the matrix...");
        var svd = matrix.Svd();
        Console.WriteLine("\nSingular Values:");
        Console.WriteLine(svd.S.ToString());

        Console.WriteLine("Selecting the number of singular values we'll retain for our lower-rank approximation...");
        int lowRank = 5;

        Console.WriteLine("Extracting the first 'lowRank' columns from the U matrix of the SVD...");
        var uMatrix = svd.U.SubMatrix(0, 100, 0, lowRank); // Adjusted size for 100x100 matrix

        Console.WriteLine("Extracting the first 'lowRank' rows from the V* matrix of the SVD...");
        var vTMatrix = svd.VT.SubMatrix(0, lowRank, 0, 100); // Adjusted size for 100x100 matrix

        Console.WriteLine("Creating a diagonal matrix with the first 'lowRank' singular values...");
        var sMatrix = Matrix<double>.Build.DenseDiagonal(lowRank, lowRank, i => svd.S[i]);

        Console.WriteLine("\nU Matrix (first 5 columns):");
        Console.WriteLine(uMatrix.ToString());

        Console.WriteLine("\nSingular Values Matrix (50x50 diagonal):");
        Console.WriteLine(sMatrix.ToString());

        Console.WriteLine("\nV Transposed Matrix (first 5 rows):");
        Console.WriteLine(vTMatrix.ToString());

        Console.WriteLine("Computing the low-rank approximation by multiplying the U, Σ, and V* matrices together...");
        var lowRankMatrix = uMatrix * sMatrix * vTMatrix;

        Console.WriteLine("\nLow Rank Approximation:");
        Console.WriteLine(lowRankMatrix.ToString());
    }

    static void RunWithoutDebug()
    {
        var rng = new System.Random();
        var matrix = Matrix<double>.Build.Dense(100, 100, (i, j) => rng.NextDouble());

        var svd = matrix.Svd();

        int lowRank = 5;

        var uMatrix = svd.U.SubMatrix(0, 100, 0, lowRank); // Adjusted size for 100x100 matrix
        var vTMatrix = svd.VT.SubMatrix(0, lowRank, 0, 100); // Adjusted size for 100x100 matrix
        var sMatrix = Matrix<double>.Build.DenseDiagonal(lowRank, lowRank, i => svd.S[i]);

        var lowRankMatrix = uMatrix * sMatrix * vTMatrix;
    }
}
