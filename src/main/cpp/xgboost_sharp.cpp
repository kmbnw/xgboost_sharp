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

#include <iostream>
#include <cstdlib>
#include "rabit/c_api.h"
#include "xgboost/c_api.h"

// these links were eminently helpful:
// http://stackoverflow.com/questions/36071672/using-xgboost-in-c
// https://stackoverflow.com/questions/38314092/reading-xgboost-model-in-c

class XGBoostWrapper {
    public:
        XGBoostWrapper();
        ~XGBoostWrapper();
        // xgboost expects a flat array
        void fit(const float Xs[], const float Ys[], unsigned int rows, unsigned int cols);
        void predict(const float Xs[], const float** YHats, unsigned int rows, unsigned int cols);
    private:
        BoosterHandle m_booster;
};

// Create an XGBoost handle
XGBoostWrapper::XGBoostWrapper() {
    m_booster = new BoosterHandle();
}

// Delete the XGBoost handle
XGBoostWrapper::~XGBoostWrapper() {
    if (m_booster) {
        XGBoosterFree(m_booster);
    }
}

void XGBoostWrapper::fit(const float Xs[], const float Ys[], unsigned int rows, unsigned int cols) {
    // convert to DMatrix
    DMatrixHandle h_train[1];
    XGDMatrixCreateFromMat((float *) Xs, rows, cols, -1, &h_train[0]);

    // load the labels
    XGDMatrixSetFloatInfo(h_train[0], "label", Ys, rows);

    // read back the labels, just a sanity check
    /*bst_ulong bst_result;
    const float *out_floats;
    XGDMatrixGetFloatInfo(h_train[0], "label" , &bst_result, &out_floats);
    for (unsigned int i=0;i<bst_result;i++)
        std::cout << "label[" << i << "]=" << out_floats[i] << std::endl;
    */

    // create the booster and load some parameters
    XGBoosterCreate(h_train, 1, &m_booster);
    XGBoosterSetParam(m_booster, "booster", "gbtree");
    XGBoosterSetParam(m_booster, "objective", "reg:linear");
    XGBoosterSetParam(m_booster, "max_depth", "5");
    XGBoosterSetParam(m_booster, "eta", "0.1");
    XGBoosterSetParam(m_booster, "min_child_weight", "1");
    XGBoosterSetParam(m_booster, "subsample", "0.5");
    XGBoosterSetParam(m_booster, "colsample_bytree", "1");
    XGBoosterSetParam(m_booster, "num_parallel_tree", "1");

    // perform 200 learning iterations
    for (unsigned int iter=0; iter<200; iter++) {
        XGBoosterUpdateOneIter(m_booster, iter, h_train[0]);
    }

    // free xgboost internal structures
    XGDMatrixFree(h_train[0]);
}

void XGBoostWrapper::predict(const float Xs[], const float** Yhats, unsigned int rows, unsigned int cols) {
    DMatrixHandle h_test;
    XGDMatrixCreateFromMat((float *) Xs, rows, cols, -1, &h_test);
    bst_ulong out_len;
    XGBoosterPredict(m_booster, h_test, 0, 0, &out_len, Yhats);

    // free xgboost internal structures
    XGDMatrixFree(h_test);
}

int main() {
    // create the train data
    const unsigned int cols = 3;
    const unsigned int rows = 5;

    float Xs[rows * cols];
    for(unsigned int i = 0; i < rows; ++i)
    {
        for(unsigned int j = 0; j < cols; ++j)
        {
            Xs[i * cols + j] = (i + 1) * (j + 1);
        }
    }

    float Ys[rows];

    for (unsigned int i = 0; i < rows; i++) {
        Ys[i] = 1 + i * i * i;
    }

    XGBoostWrapper fitter;
    fitter.fit(Xs, Ys, rows, cols);

    const unsigned int sample_rows = 5;
    float test[sample_rows * cols];
    for (unsigned int i = 0;i < sample_rows; i++) {
        for (unsigned int j = 0;j < cols; j++) {
            test[i * cols + j] = (i + 1) * (j + 1);
        }
    }

    const float* Yhats; //[sample_rows];
    fitter.predict(test, &Yhats, sample_rows, cols);

    for (unsigned int i = 0;i < sample_rows; i++) {
        std::cout << "prediction[" << i << "]=" << Yhats[i] << std::endl;
    }

    free((void *)Yhats);

    return 0;
}
