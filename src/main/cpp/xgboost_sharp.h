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
#include <map>
#include "xgboost/c_api.h"

#ifndef KMBNW_XGBOOST_SHARP_H
#define KMBNW_XGBOOST_SHARP_H

namespace xgboostsharp {
    class XGBoostWrapper {
        public:
            XGBoostWrapper(unsigned int num_trees);
            XGBoostWrapper(std::string model_file);
            ~XGBoostWrapper();
            XGBoostWrapper(const XGBoostWrapper& other) = delete;
            XGBoostWrapper& operator=(const XGBoostWrapper& other) = delete;
            // xgboost expects a flattened array of Xs
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
                XGDMatrixWrapper(const XGDMatrixWrapper& other) = delete;
                XGDMatrixWrapper& operator=(const XGDMatrixWrapper& other) = delete;
                ~XGDMatrixWrapper();

                DMatrixHandle dmatrix[1];
        };
}
#endif //KMBNW_XGBOOST_SHARP_H
