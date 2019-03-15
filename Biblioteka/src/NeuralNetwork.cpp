//
// Created by kamil on 3/14/19.
//

#include <ctime>
#include <boost/numeric/ublas/io.hpp>
#include <iostream>
#include <cmath>
#include "Point.h"
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

    randomizeMatrix(weight_ih,-1,1);
    randomizeMatrix(weight_ho,-1,1);
    randomizeMatrix(bias_o,-1,1);
    randomizeMatrix(bias_h,-1,1);
}

matrix<double> NeuralNetwork::feedForward(matrix<double> input) {

    matrix<double> hidden(weight_ih.size1(),input.size2());
    //multiplying weight array  between input and hidden by inputs
    hidden = prod(weight_ih, input);
    //adding to product  of multiplication bias weight matrix
    hidden += bias_h;
    //activation function in use
    for(int i = 0; i < hidden.size1(); i++){
        for(int j = 0; j < hidden.size2(); j++){
            hidden(i,j) = sigmoidFunction(hidden(i,j));
        }
    }

    matrix<double> output(weight_ho.size1(),hidden.size2());
    //multiplying weight array  between hidden and output by inputs
    output = prod(weight_ho, hidden);
    //adding to product  of multiplication bias weight matrix
    output += bias_o;
    //activation function in use
    for(int i = 0; i < output.size1(); i++){
        for(int j = 0; j < output.size2(); j++){
            output(i,j) = sigmoidFunction(output(i,j));
        }
    }
    return output;
}

double NeuralNetwork::sigmoidFunction(double x) {
    //activation function
    return 1/(1+exp(-x));
}

void NeuralNetwork::randomizeMatrix(matrix<double> & input, int minV, int maxV) {

    for(int i = 0; i < input.size1(); i++){
        for(int j = 0; j < input.size2(); j++){
            input(i,j) = ((double(rand())/double(RAND_MAX))*(maxV - minV))+minV;
        }
    }
}

void NeuralNetwork::train(point_ptr point) {

    matrix<double> output = feedForward(point->getInputs());
    matrix<double> error(output.size1(),output.size2());

    //Calculate error
    //ERROR = ANSWER - OUTPUT
    error = point->getAnswer() - output;
    //std::cout << error << std::endl;
    //std::cout << output << std::endl;
    //std::cout << point->getAnswer() << std::endl;
}
