using System;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using System.Linq;

public static class XGBoostAPI {
    [DllImport("libxgboost_sharp.so", CharSet = CharSet.Auto)]
    [return: MarshalAs(UnmanagedType.I1)]
    private static extern void DeleteBooster(IntPtr xgbPtr);

    [DllImport("libxgboost_sharp.so", CharSet = CharSet.Auto)]
    [return: MarshalAs(UnmanagedType.I1)]
    private static extern IntPtr CreateBooster(
        uint numTrees,
        Dictionary<string, string> boosterParams);

    [DllImport("libxgboost_sharp.so", CharSet = CharSet.Auto)]
    [return: MarshalAs(UnmanagedType.I1)]
    private static extern void Fit(
        IntPtr xgbPtr,
        float[] Xs,
        float[] Ys,
        uint rows,
        uint cols);

    [DllImport("libxgboost_sharp.so", CharSet = CharSet.Auto)]
    [return: MarshalAs(UnmanagedType.I1)]
    private static extern void Predict(
        IntPtr xgbPtr,
        float[] Xs,
        float[] Yhats,
        uint rows,
        uint cols);

    public static void Main(string[] args)
    {
        IntPtr booster = IntPtr.Zero;
        
        try 
        {
            booster = CreateBooster(200, new Dictionary<string, string>());
        }
        finally
        {
            DeleteBooster(booster);
        }
        Console.WriteLine("x");
    }
}
