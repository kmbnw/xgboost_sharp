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
#include <cstring>
#include <map>
#include "rabit/c_api.h"
#include "xgboost/c_api.h"

// these links were eminently helpful:
// http://stackoverflow.com/questions/36071672/using-xgboost-in-c
// https://stackoverflow.com/questions/38314092/reading-xgboost-model-in-c

class XGBoostWrapper {
    public:
        XGBoostWrapper(unsigned int num_trees);
        XGBoostWrapper(std::string model_file);
        ~XGBoostWrapper();
        // xgboost expects a flat array
        void fit(const float Xs[], const float Ys[], unsigned int rows, unsigned int cols);
        void predict(const float Xs[], float* Yhats, unsigned int rows, unsigned int cols);
        void save(std::string outfile);
        void set_param(std::string param_name, std::string param_value);

    private:
        BoosterHandle _h_booster;
        // number of boosting rounds
        unsigned int _num_trees;
        std::map<std::string, std::string> _model_params;
};

// wrap XGDMatrix for nice RAII behavior
class XGDMatrixWrapper {
    public:
        XGDMatrixWrapper(
                const float Xs[],
                const float Ys[],
                unsigned int rows,
                unsigned int cols);
        XGDMatrixWrapper(
                const float Xs[],
                unsigned int rows,
                unsigned int cols);
        ~XGDMatrixWrapper();

        DMatrixHandle dmatrix[1];
};

XGDMatrixWrapper::XGDMatrixWrapper(
        const float Xs[],
        const float Ys[],
        unsigned int rows,
        unsigned int cols) {
    XGDMatrixCreateFromMat((float *) Xs, rows, cols, -1, &dmatrix[0]);
    XGDMatrixSetFloatInfo(dmatrix[0], "label", Ys, rows);
}

XGDMatrixWrapper::XGDMatrixWrapper(
        const float Xs[],
        unsigned int rows,
        unsigned int cols) {
    XGDMatrixCreateFromMat((float *) Xs, rows, cols, -1, &dmatrix[0]);
}

XGDMatrixWrapper::~XGDMatrixWrapper() {
    XGDMatrixFree(dmatrix[0]);
}

// Create an XGBoost handle
XGBoostWrapper::XGBoostWrapper(unsigned int num_trees) {
    _num_trees = num_trees;
    _model_params.clear();
}

XGBoostWrapper::XGBoostWrapper(std::string model_file) {
    _h_booster = new BoosterHandle();
    XGBoosterCreate(0, 0, &_h_booster);
    XGBoosterLoadModel(_h_booster, model_file.c_str());
}

// Delete the XGBoost handle
XGBoostWrapper::~XGBoostWrapper() {
    XGBoosterFree(_h_booster);
}

void XGBoostWrapper::fit(const float Xs[], const float Ys[], unsigned int rows, unsigned int cols) {
    XGDMatrixWrapper train_mat(Xs, Ys, rows, cols);

    // create the booster and load some parameters
    XGBoosterCreate(train_mat.dmatrix, 1, &_h_booster);

    for (std::map<std::string, std::string>::iterator it = _model_params.begin();
         it != _model_params.end();
         ++it) {
        XGBoosterSetParam(_h_booster, it->first.c_str(), it->second.c_str());
    }

    for (unsigned int iter = 0; iter < _num_trees; iter++) {
        XGBoosterUpdateOneIter(_h_booster, iter, train_mat.dmatrix[0]);
    }
}

void XGBoostWrapper::predict(const float Xs[], float* Yhats, unsigned int rows, unsigned int cols) {
    XGDMatrixWrapper test_mat(Xs, rows, cols);
    bst_ulong out_len;
    const float* f;
    XGBoosterPredict(_h_booster, test_mat.dmatrix[0], 0, 0, &out_len, &f);

    std::memcpy(Yhats, f, sizeof(float) * out_len);
}

void XGBoostWrapper::save(std::string model_file) {
    XGBoosterSaveModel(_h_booster, model_file.c_str());
}

void XGBoostWrapper::set_param(std::string param_name, std::string param_value) {
    _model_params[param_name] = param_value;
}

extern "C" {
    XGBoostWrapper* CreateBooster(int num_trees) {
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
        pBooster->fit(Xs, Ys, rows, cols);
    }

    void Predict(
            XGBoostWrapper* pBooster,
            const float Xs[],
            float* Yhats,
            unsigned int rows,
            unsigned int cols) {
        pBooster->predict(Xs, Yhats, rows, cols);
    }

    
    XGBoostWrapper* LoadModel(const char* model_file) {
        return new XGBoostWrapper(std::string(model_file));
    }
    
    void SaveModel(XGBoostWrapper* pBooster, const char* outfile) {
        pBooster->save(std::string(outfile));
    }
    
    void SetParam(XGBoostWrapper* pBooster, const char* param_name, const char* param_value) {
        pBooster->set_param(std::string(param_name), std::string(param_value));
    }
}
