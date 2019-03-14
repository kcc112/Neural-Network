//
// Created by kamil on 3/14/19.
//

#ifndef NEURALNETWORK_NEURALNETWORK_H
#define NEURALNETWORK_NEURALNETWORK_H


class NeuralNetwork {

    int numInput;
    int numHidden;
    int numOutput;

public:
    NeuralNetwork(int numInput, int numHidden, int numOutput);
    ~NeuralNetwork() = default;

};


#endif //NEURALNETWORK_NEURALNETWORK_H
