//
// Created by kamil on 3/14/19.
//

#include <ctime>
#include <boost/numeric/ublas/io.hpp>
#include <iostream>
#include <cmath>
#include "Point.h"
#include "NeuralNetwork.h"

//create a three layer neural network
NeuralNetwork::NeuralNetwork(unsigned int inputNodes, unsigned int hiddenNodes, unsigned int outputNodes, unsigned int hiddenSize,  unsigned int period, unsigned int trainingSize, double momentum, double lRate, int biaS) {

    srand(time(nullptr));
    this->inputNodes = inputNodes;
    this->hiddenNodes = hiddenNodes;
    this->outputNodes = outputNodes;
    this->hiddenSize = hiddenSize;
    this->biaSS = biaS;
    this->lRate = lRate;
    this->momentum = momentum;

    difPow.reserve(period * trainingSize);
    weights.reserve(hiddenSize + 1);
    if(biaSS == 1) {
        biasWeights.reserve(hiddenSize + 1);
    }
    momentumWeights.reserve(hiddenSize + 1);

    if(hiddenSize > 2) {
        if(biaSS == 1) {
            biasWeights[0] = matrix<double>(hiddenNodes, 1);
        }
        weights[0] = matrix<double>(hiddenNodes, inputNodes);
        momentumWeights[0] = matrix<double>(hiddenNodes, inputNodes);

        for(int i = 0; i < hiddenSize - 1; i++) {
            if(biaSS == 1) {
                biasWeights[i + 1] = matrix<double>(hiddenNodes, 1);
            }
            weights[i + 1] = matrix<double >(hiddenNodes, hiddenNodes);
            momentumWeights[i + 1] = matrix<double >(hiddenNodes, hiddenNodes);
        }

        momentumWeights[hiddenSize] = matrix<double>(outputNodes, hiddenNodes);
        weights[hiddenSize] = matrix<double>(outputNodes, hiddenNodes);
        if(biaSS == 1) {
            biasWeights[hiddenSize] = matrix<double>(outputNodes, 1);
        }

    } else if(hiddenSize == 1) {
        weights[0] = matrix<double>(hiddenNodes, inputNodes);
        weights[1] = matrix<double>(outputNodes, hiddenNodes);
        momentumWeights[0] = matrix<double>(hiddenNodes, inputNodes);
        momentumWeights[1] = matrix<double>(outputNodes, hiddenNodes);
        if(biaSS == 1) {
            biasWeights[0] = matrix<double>(hiddenNodes, 1);
            biasWeights[1] = matrix<double>(outputNodes, 1);
        }


    } else if(hiddenSize == 2) {
        weights[0] = matrix<double>(hiddenNodes, inputNodes);
        weights[1] = matrix<double>(hiddenNodes, hiddenNodes);
        weights[2] = matrix<double>(outputNodes, hiddenNodes);
        momentumWeights[0] = matrix<double>(hiddenNodes, inputNodes);
        momentumWeights[1] = matrix<double>(hiddenNodes, hiddenNodes);
        momentumWeights[2] = matrix<double>(outputNodes, hiddenNodes);
        if(biaSS == 1) {
            biasWeights[0] = matrix<double>(hiddenNodes, 1);
            biasWeights[1] = matrix<double>(hiddenNodes, 1);
            biasWeights[2] = matrix<double>(outputNodes, 1);
        }
    }

    for(int i = 0; i < hiddenSize + 1; i++) {
        randomizeMatrix(weights[i], -0.5, 0.5);
        if(biaSS == 1) {
            randomizeMatrix(biasWeights[i], -0.5, 0.5);
        }
    }
}

matrix<double> NeuralNetwork::calculateHiddenNew(const matrix<double> & input, int number) {
    matrix<double> hidden(weights[number].size1(), input.size2());
    hidden = prod(weights[number], input);
    if(biaSS == 1) {
        hidden += biasWeights[number];
    }
    for(int i = 0; i < hidden.size1(); i++){
        for(int j = 0; j < hidden.size2(); j++){
            hidden(i,j) = sigmoidFunction(hidden(i,j));
        }
    }
    return hidden;
}

matrix<double> NeuralNetwork::calculateOutputNew(const matrix<double> & hidden) {
    matrix<double> output(weights[hiddenSize].size1(), hidden.size2());
    output = prod(weights[hiddenSize], hidden);
    if(biaSS == 1) {
        output += biasWeights[hiddenSize];
    }
    for(int i = 0; i < output.size1(); i++){
        for(int j = 0; j < output.size2(); j++){
            output(i,j) = sigmoidFunction(output(i,j));
        }
    }
    return output;
}

matrix<double> NeuralNetwork::feedForwardNew(const matrix<double> & input) {

    matrix<double> hidden = calculateHiddenNew(input, 0);
    matrix<double> output;

    for(int i = 0; i < hiddenSize - 1; i++) {
        hidden = calculateHiddenNew(hidden, i + 1);
    }
    output = calculateOutputNew(hidden);
    return output;
}

double NeuralNetwork::sigmoidFunction(double x) {
    //activation function
    return 1/(1+exp(-x));
}


double NeuralNetwork::dsigmoidFunction(double x) {
    // return sigmoid(x) * (1 - sigmoid(x);
    return x * (1 - x);
}


void NeuralNetwork::randomizeMatrix(matrix<double> & input, double minV, double maxV) {

    for(int i = 0; i < input.size1(); i++){
        for(int j = 0; j < input.size2(); j++){
            input(i,j) = ((double(rand())/double(RAND_MAX))*(maxV - minV))+minV;
        }
    }
}

void NeuralNetwork::cost(const matrix<double> & input) {
    double sum = 0;
    for (int i = 0; i < input.size1(); i++) {
        sum += pow(input(i,0),2);
    }
    difPow.push_back(sum);
}


void NeuralNetwork::trainNew(const point_ptr & point) {

    matrix<double> input = point->getInputs();
    std::vector<matrix<double>> hidden(hiddenSize);
    std::vector<matrix<double>> hiddenT(hiddenSize);
    std::vector<matrix<double>> hiddenGradient(hiddenSize);


    for(int i = 0; i < hiddenSize; i++) {
        hidden[i] = matrix<double>(hiddenNodes, 1);
        hiddenT[i] = matrix<double>(1, hiddenNodes);
        if(i == 0) {
            hidden[i] = prod(weights[i], input);
        } else {
            hidden[i] = prod(weights[i], hidden[i - 1]);
        }
        if(biaSS == 1) {
            hidden[i] += biasWeights[i];
        }
        for(int z = 0; z < hidden[i].size1(); z++){
            for(int j = 0; j < hidden[i].size2(); j++){
                hidden[i](z,j) = sigmoidFunction(hidden[i](z,j));
            }
        }
    }


    matrix<double> output = calculateOutputNew(hidden[hiddenSize - 1]);

    matrix<double> outputErrors = point->getAnswer() - output;
    cost(outputErrors);

    matrix<double> gradient = output;
    hiddenGradient = hidden;


    for(int i = 0; i < gradient.size1(); i++){
        for(int j = 0; j < gradient.size2(); j++){
            gradient(i,j) = dsigmoidFunction(gradient(i,j));
            gradient(i,j) *= outputErrors(i,j);
        }
    }


    for(int i = 0; i < hiddenSize; i++) {
        hiddenT[i] = trans(hidden[i]);
    }

    matrix<double> weight_ho_deltas = prod(gradient, hiddenT[hiddenSize - 1]);
    matrix<double> weight_hot = trans(weights[hiddenSize]);
    matrix<double> hiddenErrors = prod(weight_hot, gradient);
    weight_ho_deltas *= lRate;
    weights[hiddenSize] += weight_ho_deltas + momentum * momentumWeights[hiddenSize];
    if(biaSS == 1) {
        biasWeights[hiddenSize] += gradient;
    }
    momentumWeights[hiddenSize] = weight_ho_deltas + momentum * momentumWeights[hiddenSize];


    for(int i = hiddenSize; i > 0; i--) {
        for(int z = 0; z < hiddenGradient[i - 1].size1(); z++){
            for(int j = 0; j < hiddenGradient[i - 1].size2(); j++){
                hiddenGradient[i - 1](z,j) = dsigmoidFunction(hiddenGradient[i - 1](z,j));
                hiddenGradient[i - 1](z,j) *= hiddenErrors(z,j);
            }
        }


        if(i - 1 != 0) {
            hiddenErrors += prod(trans(weights[i - 1]), hiddenGradient[i - 1]);
        }
        hiddenGradient[i - 1] *= lRate;
        matrix<double> weight_ih_deltas;
        if(i - 1 == 0) {
            weight_ih_deltas = prod(hiddenGradient[i - 1], trans(point->getInputs()));
        } else {
            weight_ih_deltas = prod(hiddenGradient[i - 1], hiddenT[i - 2]);
        }
        weights[i - 1] += weight_ih_deltas + momentum * momentumWeights[i - 1];
        if(biaSS == 1) {
            biasWeights[i - 1] += hiddenGradient[i - 1];
        }
        momentumWeights[i - 1] = weight_ih_deltas + momentum * momentumWeights[i - 1];
    }
}

std::vector<double> NeuralNetwork::getDifPow() {
    return difPow;
}
