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
#include <map>
#include "rabit/c_api.h"
#include "xgboost/c_api.h"

// these links were eminently helpful:
// http://stackoverflow.com/questions/36071672/using-xgboost-in-c
// https://stackoverflow.com/questions/38314092/reading-xgboost-model-in-c

class XGBoostWrapper {
    public:
        XGBoostWrapper(unsigned int num_trees);
        ~XGBoostWrapper();
        // xgboost expects a flat array
        void fit(const float Xs[], const float Ys[], unsigned int rows, unsigned int cols);
        void predict(const float Xs[], const float** Yhats, unsigned int rows, unsigned int cols);
    private:
        BoosterHandle _h_booster;
        // number of boosting rounds
        unsigned int _num_trees;
};

// Create an XGBoost handle
XGBoostWrapper::XGBoostWrapper(unsigned int num_trees) {
    _h_booster = new BoosterHandle();
    _num_trees = num_trees;
}

// Delete the XGBoost handle
XGBoostWrapper::~XGBoostWrapper() {
    if (_h_booster) {
        XGBoosterFree(_h_booster);
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
    XGBoosterCreate(h_train, 1, &_h_booster);

    std::map<std::string, std::string> booster_params;
    booster_params["booster"] = "gbtree";
    booster_params["objective"] = "reg:linear";
    booster_params["max_depth"] = "5";
    booster_params["eta"] = "0.1";
    booster_params["min_child_weight"] = "1";
    booster_params["subsample"] = "0.5";
    booster_params["colsample_bytree"] = "1";
    booster_params["num_parallel_tree"] = "1";

    std::map<std::string, std::string>::iterator it;
    for (it = booster_params.begin(); it != booster_params.end(); ++it) {
        std::cout << it->first << ", " << it->second << '\n';
        XGBoosterSetParam(_h_booster, it->first.c_str(), it->second.c_str());
    }

    for (unsigned int iter = 0; iter < _num_trees; iter++) {
        XGBoosterUpdateOneIter(_h_booster, iter, h_train[0]);
    }

    // free xgboost internal structures
    XGDMatrixFree(h_train[0]);
}

void XGBoostWrapper::predict(const float Xs[], const float** Yhats, unsigned int rows, unsigned int cols) {
    DMatrixHandle h_test;
    XGDMatrixCreateFromMat((float *) Xs, rows, cols, -1, &h_test);
    bst_ulong out_len;
    XGBoosterPredict(_h_booster, h_test, 0, 0, &out_len, Yhats);

    // free xgboost internal structures
    XGDMatrixFree(h_test);
}

int main() {
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

    XGBoostWrapper fitter(200);
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

extern "C" {
    XGBoostWrapper* CreateBooster(unsigned int num_trees) {
        return new XGBoostWrapper(num_trees);
    }

    void DeleteBooster(XGBoostWrapper* pBooster) {
        if (pBooster) {
            delete pBooster;
        }
    }

    void Fit(
            XGBoostWrapper* pBooster,
            const float Xs[],
            const float Ys[],
            unsigned int rows,
            unsigned int cols) {
        return pBooster->fit(Xs, Ys, rows, cols);
    }

    void Predict(
          XGBoostWrapper* pBooster,
          const float Xs[],
          const float** Yhats,
          unsigned int rows,
          unsigned int cols) {
      return pBooster->predict(Xs, Yhats, rows, cols);
    }
}
