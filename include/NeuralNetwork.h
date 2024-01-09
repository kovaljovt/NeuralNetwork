#ifndef NEURALNETWORK_NEURALNETWORK_H
#define NEURALNETWORK_NEURALNETWORK_H

#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <vector>

typedef float Scalar;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::RowVectorXf RowVector;
typedef Eigen::VectorXf ColVector;

class NeuralNetwork {
public:
    std::vector<RowVector *> neuronLayers;
    std::vector<RowVector *> cacheLayers;
    std::vector<RowVector *> deltas;
    std::vector<Matrix *> weights;
    std::vector<uint> topology;
    Scalar learningRate;

    NeuralNetwork(std::vector<uint> topology, Scalar learningRate = Scalar(0.005));

    void propagateForward(RowVector &input);
    void propagateBackward(RowVector &output);
    void calculateErrors(RowVector &output);
    void updateWeights();
    void train(std::vector<RowVector *> data);
};


#endif //NEURALNETWORK_NEURALNETWORK_H
