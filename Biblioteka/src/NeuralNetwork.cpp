//
// Created by kamil on 3/14/19.
//

#include <ctime>
#include <boost/numeric/ublas/io.hpp>
#include <iostream>
#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(unsigned int inputNodes, unsigned int hiddenNodes, unsigned int outputNodes) {
    srand(time(NULL));
    this->inputNodes = inputNodes;
    this->hiddenNodes = hiddenNodes;
    this->outputNodes = outputNodes;
    weight_ih = matrix<double>(hiddenNodes,inputNodes);
    weight_ho = matrix<double>(outputNodes,hiddenNodes);
    bias_o = matrix<double>(outputNodes,1);
    bias_h = matrix<double>(hiddenNodes,1);

    for(int i = 0; i < hiddenNodes; i++){
        for(int j = 0; j < inputNodes; j++){
            weight_ih(i,j) = ((double(rand())/double(RAND_MAX))*(2))-1;
        }
    }

    for(int i = 0; i < outputNodes; i++){
        for(int j = 0; j < hiddenNodes; j++){
            weight_ho(i,j) = ((double(rand())/double(RAND_MAX))*(2))-1;
        }
    }
}

int NeuralNetwork::feedForward(matrix<double> input) {
    matrix<double> hidden(hiddenNodes,input.size2());
    hidden = prod(weight_ih, input);
    hidden += bias_h;
    std::cout <<  hidden << std::endl;
    return 0;
}
