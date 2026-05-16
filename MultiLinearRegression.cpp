/*
    predicts grade based on sleep, studying, outdoor activity, and time gaming
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
using namespace std;

// references to vector instead of copying them, they cannot be changed here because it is a const
double dotProduct(const vector<double>& a, const vector<double>& b) {
    double total = 0;
    // assumes a and b are same size
    for (int i = 0; i < a.size(); i++) {
        total += a[i] * b[i];
    }
    return total;
}

class multiLinearRegressionModel {
    private:
        vector<double> weights;
        double bias = 0;
        vector<double> featureMins;
        vector<double> featureMaxes;
        
        vector<vector<double>> normalizeTrainingInputs(const vector<vector<double>>& unnormalizedInputs) {
            vector<vector<double>> normalizedInputs = unnormalizedInputs;

            int n = normalizedInputs.size();
            int numFeatures = normalizedInputs[0].size();
            weights.assign(numFeatures, 0.0);

            featureMins.resize(numFeatures);
            featureMaxes.resize(numFeatures);

            for (int featureIdx = 0; featureIdx < numFeatures; featureIdx++) {
                double minValue = 1e9;
                double maxValue = -1e9;

                // find min/max
                for (int dataPointIdx = 0; dataPointIdx < n; dataPointIdx++) {
                    double value = normalizedInputs[dataPointIdx][featureIdx];

                    if (value < minValue) {
                        minValue = value;
                    }

                    if (value > maxValue) {
                        maxValue = value;
                    }
                }

                featureMins[featureIdx] = minValue;
                featureMaxes[featureIdx] = maxValue;

                // normalize column
                for (int dataPointIdx = 0; dataPointIdx < n; dataPointIdx++) {
                    normalizedInputs[dataPointIdx][featureIdx] =
                        (normalizedInputs[dataPointIdx][featureIdx] - minValue)
                        / (maxValue - minValue);
                }
            }

            return normalizedInputs;
        }

        vector<double> normalizeOneInput(const vector<double>& unnormalizedInput) {
            vector<double> normalizedInput = unnormalizedInput;

            for (int featureIdx = 0; featureIdx < normalizedInput.size(); featureIdx++) {
                normalizedInput[featureIdx] =
                    (normalizedInput[featureIdx] - featureMins[featureIdx])
                    / (featureMaxes[featureIdx] - featureMins[featureIdx]);
            }

            return normalizedInput;
        }

    public:
        void fit(const vector<vector<double>>& unnormalizedInputs, const vector<double>& expectedOutputs) {
            vector<vector<double>> normalizedInputs = normalizeTrainingInputs(unnormalizedInputs);
            int n = normalizedInputs.size();

            // training values
            int iterations = 7.5e+3;
            double learningRate = 3e-3;

            double currentLoss = 0;
            double error;

            // initial loss
            double totalError = 0;

            for (int dataPointIdx = 0; dataPointIdx < n; dataPointIdx++) {
                // y_hat = w[0] * x[0] + ... + w[n] * x[n] + b but dot product is more readable and efficient
                double pred = dotProduct(weights, normalizedInputs[dataPointIdx]) + bias;
                error = pred - expectedOutputs[dataPointIdx];
                totalError += error * error; // sum of squared errors (SSE) or Σ(error²)
            }

            currentLoss = totalError / (2 * n);


            // training loop
            for (int epoch = 0; epoch < iterations; epoch++) {
                // stopping early if loss is already low enough
                if (currentLoss < 1e-6) break;

                // resetting variables
                double totalError = 0;
                double biasChange = 0;
                vector<double> weightChanges(weights.size(), 0.0);

                for (int dataPointIdx = 0; dataPointIdx < n; dataPointIdx++) {
                    // y_hat = w[0] * x[0] + ... + w[n] * x[n] + b but dot product is more readable and efficient
                    double pred = dotProduct(weights, normalizedInputs[dataPointIdx]) + bias;
                    double error = pred - expectedOutputs[dataPointIdx];

                    totalError += error * error; // sum of squared errors (SSE) or Σ(error²)
                    biasChange += error;

                    // calculates how much to adjust each weight to reduce the error for a specific point (dataPointIdx)
                    for (int weightChangeIdx = 0; weightChangeIdx < weights.size(); weightChangeIdx++) {
                        weightChanges[weightChangeIdx] += error * normalizedInputs[dataPointIdx][weightChangeIdx];
                    }
                }

                currentLoss = totalError / (2 * n);

                // adjust weights and bias
                biasChange /= n;
                for (int weightChangeIdx = 0; weightChangeIdx < weights.size(); weightChangeIdx++) {
                    weightChanges[weightChangeIdx] /= n;
                }

                // applying changes
                bias -= learningRate * biasChange;
                for (int weightAdjustmentIdx = 0; weightAdjustmentIdx < weights.size(); weightAdjustmentIdx++) {
                    weights[weightAdjustmentIdx] -= learningRate * weightChanges[weightAdjustmentIdx];
                }

                // learning rate decay
                if (epoch > 0 && epoch % 1000 == 0) learningRate *= 0.95;
            }
        }

        double predict(const vector<double>& features) {
            vector<double> normalizedInput = normalizeOneInput(features);
            return clamp(dotProduct(weights, normalizedInput) + bias, 0.0, 100.0);
        }
};

int main() {
    vector<vector<double>> inputs = {
        {8, 2, 1, 4},      // decent habits, moderate gaming
        {7, 4, 1, 2},      // balanced
        {6, 6, 0.5, 1},    // high study
        {5, 7, 0.5, 0.5},  // intense studying
        {9, 1, 2, 5},      // great sleep/activity but distracted
        {7, 5, 1.5, 1},    // healthy balance
        {4, 2, 0.5, 6},    // poor habits
        {8, 3, 1, 3},      // average-good student
        {8, 5, 1, 2},      // strong balance
        {6, 3, 2, 3},      // decent but inconsistent
        {7, 2, 2, 4},      // relaxed but functional
        {9, 4, 2, 1},      // very healthy/studious
        {5, 5, 1, 4},      // studies but poor balance
        {6, 1, 1, 5},      // distracted
        {8, 6, 1, 1},      // excellent discipline
        {7, 3, 0.5, 2},    // normal student
        {6, 4, 1.5, 2},    // decent consistency
        {9, 2, 2, 2},      // healthy but not obsessive
        {5, 3, 0.5, 5},    // mediocre habits
        {8, 4, 2, 3}       // balanced lifestyle
    };

    vector<double> expectedOutputs = {
        82, 86, 92, 94, 78, 89, 60, 84, 91, 80, 79, 93, 81, 68, 96, 83, 85, 87, 70, 88
    };

    multiLinearRegressionModel model;
    model.fit(inputs, expectedOutputs);
    cout << model.predict({8, 3, 1, 3}) << "\n";

    return 0;
}
