using System;
using MathNet.Numerics.LinearAlgebra;

class Program
{
    static void Main(string[] args)
    {
        // Create a 10x10 sample matrix
        Matrix<double> A = Matrix<double>.Build.DenseOfArray(new double[,] {
            { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
            { 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 },
            { 21, 22, 23, 24, 25, 26, 27, 28, 29, 30 },
            { 31, 32, 33, 34, 35, 36, 37, 38, 39, 40 },
            { 41, 42, 43, 44, 45, 46, 47, 48, 49, 50 },
            { 51, 52, 53, 54, 55, 56, 57, 58, 59, 60 },
            { 61, 62, 63, 64, 65, 66, 67, 68, 69, 70 },
            { 71, 72, 73, 74, 75, 76, 77, 78, 79, 80 },
            { 81, 82, 83, 84, 85, 86, 87, 88, 89, 90 },
            { 91, 92, 93, 94, 95, 96, 97, 98, 99, 100 }
        });

        Console.WriteLine("Original matrix A:");
        Console.WriteLine(A);

        // Perform SVD
        var svd = A.Svd(true);

        // Determine rank for low-rank approximation
        int lowRank = 2;

        // Compute low-rank approximation
        var sMatrix = Matrix<double>.Build.Dense(lowRank, lowRank, (i, j) => i == j ? svd.S[i] : 0.0);
        Matrix<double> A_approx = svd.U.SubMatrix(0, A.RowCount, 0, lowRank) *
                                  sMatrix *
                                  svd.VT.SubMatrix(0, lowRank, 0, A.ColumnCount);

        Console.WriteLine("\nLow-rank approximation of A:");
        Console.WriteLine(A_approx);
    }
}
