using System;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using System.Linq;

public static class XGBoostAPI {
    [DllImport("libxgboost_sharp.so", CharSet = CharSet.Auto)]
    [return: MarshalAs(UnmanagedType.I1)]
    private static extern void DeleteBooster(IntPtr pXGBooster);

    [DllImport("libxgboost_sharp.so", CharSet = CharSet.Auto)]
    [return: MarshalAs(UnmanagedType.I1)]
    private static extern IntPtr CreateBooster(uint numTrees);

    [DllImport("libxgboost_sharp.so", CharSet = CharSet.Auto)]
    [return: MarshalAs(UnmanagedType.I1)]
    private static extern void Fit(
        IntPtr pXGBooster,
        float[] Xs,
        float[] Ys,
        uint rows,
        uint cols);

    [DllImport("libxgboost_sharp.so", CharSet = CharSet.Auto)]
    [return: MarshalAs(UnmanagedType.I1)]
    private static extern void Predict(
        IntPtr pXGBooster,
        float[] Xs,
        float[] Yhats,
        uint rows,
        uint cols);

    public static void Main(string[] args)
    {
        IntPtr booster = IntPtr.Zero;

        const uint cols = 3;
        const uint rows = 5;

        float[] Xs = new float[rows * cols];
        for(uint i = 0; i < rows; ++i)
        {
            for(uint j = 0; j < cols; ++j)
            {
                Xs[i * cols + j] = (i + 1) * (j + 1);
            }
        }

        float[] Ys = new float[rows];

        for (uint i = 0; i < rows; i++) 
        {
            Ys[i] = 1 + i * i * i;
        }

        const int sample_rows = 5;
        float[] test = new float[sample_rows * cols];
        for (int i = 0;i < sample_rows; i++) 
        {
            for (uint j = 0; j < cols; j++) 
            {
                test[i * cols + j] = (i + 1) * (j + 1);
            }
        }

        float[] Yhats = new float[sample_rows];
        
        try 
        {
            booster = CreateBooster(200);
            Fit(booster, Xs, Ys, rows, cols);

            // XGBoost allocates the array for us, so just grab the pointer
            //IntPtr yhatsPtr = IntPtr.Zero;
            Predict(booster, test, Yhats, sample_rows, cols);
            //Marshal.Copy(yhatsPtr, Yhats, 0, sample_rows);
            //Marshal.Free(yhatsPtr);

            for (uint i = 0;i < sample_rows; i++) {
                Console.WriteLine("prediction[" + i + "]=" + Yhats[i]);
            }
        }
        finally
        {
            DeleteBooster(booster);
        }
    }
}
