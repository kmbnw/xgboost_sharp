/*
 * Copyright 2016 Krysta M Bouzek
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
using System;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using System.Linq;

namespace kmbnw.XGBoostSharp
{
    public sealed class XGBoostAPI : IDisposable
    {
        // WARNING: do not unseal this class without fixing the Dispose() methods
        // to handle inheritance

        [DllImport("libxgboost_sharp.so", CharSet = CharSet.Auto)]
        [return: MarshalAs(UnmanagedType.I1)]
        private static extern void DeleteBooster(IntPtr pXGBooster);

        [DllImport("libxgboost_sharp.so", CharSet = CharSet.Auto)]
        [return: MarshalAs(UnmanagedType.I1)]
        private static extern IntPtr CreateBooster(uint numTrees);

        [DllImport("libxgboost_sharp.so", CharSet = CharSet.Auto)]
        [return: MarshalAs(UnmanagedType.I1)]
        private static extern void Fit(IntPtr pXGBooster, float[] Xs, float[] Ys, uint rows, uint cols);

        [DllImport("libxgboost_sharp.so", CharSet = CharSet.Auto)]
        [return: MarshalAs(UnmanagedType.I1)]
        private static extern void Predict(IntPtr pXGBooster, float[] Xs, float[] Ys, uint rows, uint cols);

        [DllImport("libxgboost_sharp.so", CharSet = CharSet.Auto)]
        [return: MarshalAs(UnmanagedType.I1)]
        private static extern IntPtr LoadModel(string outfile);

        [DllImport("libxgboost_sharp.so", CharSet = CharSet.Auto)]
        [return: MarshalAs(UnmanagedType.I1)]
        private static extern void SaveModel(IntPtr pXGBooster, string outfile);

        [DllImport("libxgboost_sharp.so", CharSet = CharSet.Auto)]
        [return: MarshalAs(UnmanagedType.I1)]
        private static extern void SetParam(IntPtr pXGBooster, string paramName, string paramValue);

        private IntPtr _xgboostPtr = IntPtr.Zero;

        /// <summary>
        /// Create a new XGBoost instance for use in building a new model.
        /// </summary>
        /// <param name="numRounds">Number of rounds of boosting (i.e. number of trees).</param>
        /// <param name="modelParams">Other XGBoost model parameters.
        /// Defaults will be used if null.</param>
        public XGBoostAPI(uint numRounds, Dictionary<string, string> modelParams)
        {
            _xgboostPtr = CreateBooster(numRounds);
            if (modelParams != null)
            {
                foreach (var kv in modelParams)
                {
                    SetParam(_xgboostPtr, kv.Key, kv.Value);
                }
            }
        }

        /// <summary>
        /// Load a model file in the native XGBoost format from disk.
        /// </summary>
        /// <param name="modelFile">The model file to load.</param>
        public XGBoostAPI(string modelFile)
        {
            _xgboostPtr = LoadModel(modelFile);
        }

        /// <summary>
        /// Fit a model which is held in the internal state of this instance.
        /// </summary>
        /// <param name="Xs">Matrix of features flattened into a 1-D array.</param>
        /// <param name="Ys">Array of response variables (i.e. prediction target).
        /// The length of this will be assumed to be the number of rows.</param>
        /// <param name="numRows">Number of rows in Xs.</param>
        /// <param name="numFeatures">Number of features (i.e. columns) in Xs.</param>
        public void Fit(float[] Xs, float[] Ys, uint numRows, uint numFeatures)
        {
            if (_isDisposed)
            {
                throw new ObjectDisposedException("XGBoost");
            }

            Fit(_xgboostPtr, Xs, Ys, numRows, numFeatures);
        }

        /// <summary>
        /// Predict response using the model held by this instance.
        /// </summary>
        /// <param name="Xs">Matrix of features flattened into a 1-D array.
        /// The features MUST be in the same order as what the model was trained on, or this
        /// will return bad results.</param>
        /// <param name="numRows">Number of rows in Xs.</param>
        /// <param name="numFeatures">Number of features (i.e. columns) in Xs.</param>
        /// <returns>Predicted response.</returns>
        public float[] Predict(float[] Xs, uint numRows, uint numFeatures)
        {
            if (_isDisposed)
            {
                throw new ObjectDisposedException("XGBoost");
            }

            float[] yhats = new float[numRows];

            Predict(_xgboostPtr, Xs, yhats, numRows, numFeatures);
            return yhats;
        }

        /// <summary>
        /// Save the model held by this instance to disk in the native XGBoost format.
        /// </summary>
        /// <param name="modelFile">File to save to.</param>
        public void Save(string modelFile)
        {
            if (_isDisposed)
            {
                throw new ObjectDisposedException("XGBoost");
            }
            if (_xgboostPtr == IntPtr.Zero)
            {
                throw new InvalidOperationException("Cannot save a model that has not been initialized");
            }
            SaveModel(_xgboostPtr, modelFile);
        }

        #region IDisposable Support
        private bool _isDisposed = false; // To detect redundant calls

        void Dispose(bool disposing)
        {
            if (!_isDisposed)
            {
                if (disposing)
                {
                    // dispose managed state (managed objects).
                }

                // free unmanaged resources (unmanaged objects) and override a finalizer below.
                // set large fields to null.
                if (_xgboostPtr != IntPtr.Zero)
                {
                    DeleteBooster(_xgboostPtr);
                    _xgboostPtr = IntPtr.Zero;
                }

                _isDisposed = true;
            }
        }

        // override a finalizer only if Dispose(bool disposing) above has code to free unmanaged resources.
        ~XGBoostAPI()
        {
           // Do not change this code. Put cleanup code in Dispose(bool disposing) above.
           Dispose(false);
         }

        // This code added to correctly implement the disposable pattern.
        public void Dispose()
        {
            // Do not change this code. Put cleanup code in Dispose(bool disposing) above.
            Dispose(true);
            // uncomment the following line if the finalizer is overridden above.
            GC.SuppressFinalize(this);
        }
        #endregion

        public static void Main(string[] args)
        {
            var modelParams = new Dictionary<string, string>();

            modelParams["booster"] = "gbtree";
            modelParams["objective"] = "reg:linear";
            modelParams["max_depth"] = "5";
            modelParams["eta"] = "0.1";
            modelParams["min_child_weight"] = "1";
            modelParams["subsample"] = "0.5";
            modelParams["colsample_bytree"] = "1";
            modelParams["num_parallel_tree"] = "1";

            using (var booster = new XGBoostAPI(200, modelParams))
            {
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

                booster.Fit(Xs, Ys, rows, cols);

                const int sample_rows = 5;
                float[] test = new float[sample_rows * cols];
                for (int i = 0;i < sample_rows; i++) 
                {
                    for (uint j = 0; j < cols; j++) 
                    {
                        test[i * cols + j] = (i + 1) * (j + 1);
                    }
                }

                float[] Yhats = booster.Predict(test, sample_rows, cols);

                for (uint i = 0;i < sample_rows; i++)
                {
                    Console.WriteLine("prediction[" + i + "]=" + Yhats[i]);
                }
            }
        }
    }
}
