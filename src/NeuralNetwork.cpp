#include "../include/NeuralNetwork.h"
#include <Eigen/Eigen>

NeuralNetwork::NeuralNetwork(std::vector<uint> topology, Scalar learningRate) {
    this->topology = topology;
    this->learningRate = learningRate;
    for (uint i = 0; i < topology.size(); i++) {
        if (i == topology.size() - 1) {
            neuronLayers.push_back(new RowVector(topology[i]));
        }
        else {
            neuronLayers.push_back(new RowVector(topology[i] + 1));
        }
        cacheLayers.push_back(new RowVector(neuronLayers.size()));
        deltas.push_back(new RowVector(neuronLayers.size()));

        // vector.back() gives the handle to recently added element
        // coeffRef gives the reference of value at that place
        // (using this as we are using pointers here)
        if (i != topology.size() - 1) {
            neuronLayers.back()->coeffRef(topology[i]) = 1.0;
            cacheLayers.back()->coeffRef(topology[i]) = 1.0;
        }

        // initialize weights matrix
        if (i > 0) {
            if (i != topology.size() - 1) {
                weights.push_back(new Matrix(topology[i - 1] + 1, topology[i] + 1));
                weights.back()->setRandom();
                weights.back()->col(topology[i]).setZero();
                weights.back()->coeffRef(topology[i - 1], topology[i]) = 1.0;
            }
            else {
                weights.push_back(new Matrix(topology[i - 1] + 1, topology[i]));
                weights.back()->setRandom();
            }
        }
    }
}

void NeuralNetwork::propagateForward(RowVector &input) {
    neuronLayers.front()->block(0, 0, 1, neuronLayers.front()->size() - 1) = input;
    for (uint i = 1; i < topology.size(); i++) {
        (*neuronLayers[i]) = (*neuronLayers[i - 1]) * (*weights[i - 1]);
            neuronLayers[i]->block(0, 0, 1, topology[i]).unaryExpr(std::ptr_fun(activationFunction));
    }
}

void NeuralNetwork::calculateErrors(RowVector &output) {
    (*deltas.back()) = output - (*neuronLayers.back());
    for (uint i = topology.size() - 2; i > 0; i--) {
        (*deltas[i]) = (*deltas[i + 1]) * (weights[i]->transpose());
    }
}
