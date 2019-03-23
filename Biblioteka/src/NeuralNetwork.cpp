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
NeuralNetwork::NeuralNetwork(unsigned int inputNodes, unsigned int hiddenNodes, unsigned int outputNodes) {

    srand(time(NULL));
    //sum of nodes in every layer
    //this->inputNodes = inputNodes;
    //this->hiddenNodes = hiddenNodes;
    //this->outputNodes = outputNodes;

    //weight matrix between input and hidden layer
    weight_ih = matrix<double>(hiddenNodes,inputNodes);
    //weight matrix between hidden and output layer
    weight_ho = matrix<double>(outputNodes,hiddenNodes);
    //bias matrix for output layer
    bias_o = matrix<double>(outputNodes,1);
    //bias matrix for hidden layer
    bias_h = matrix<double>(hiddenNodes,1);
    //learning rate
    lRate = 0.5;

    //randomizing weights and bias
    randomizeMatrix(weight_ih,-1,1);
    randomizeMatrix(weight_ho,-1,1);
    randomizeMatrix(bias_o,-1,1);
    randomizeMatrix(bias_h,-1,1);
}

matrix<double> NeuralNetwork::calculateHidden(matrix<double> input) {

    //create matrix for hidden output (h[i,j])
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
    return hidden;
}

matrix<double> NeuralNetwork::calculateOutput(matrix<double> hidden) {

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

//implementation feed forward algorithm
matrix<double> NeuralNetwork::feedForward(matrix<double> input) {

    //calculate hidden
    matrix<double> hidden = calculateHidden(input);
    //calculate output
    matrix<double> output = calculateOutput(hidden);

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


void NeuralNetwork::randomizeMatrix(matrix<double> & input, int minV, int maxV) {

    for(int i = 0; i < input.size1(); i++){
        for(int j = 0; j < input.size2(); j++){
            input(i,j) = ((double(rand())/double(RAND_MAX))*(maxV - minV))+minV;
        }
    }
}

void NeuralNetwork::train(point_ptr point) {


    //Feed forward part
    matrix<double> input = point->getInputs();
    matrix<double> hidden = calculateHidden(input);
    matrix<double> output = calculateOutput(hidden);

    //Calculate error output and hidden
    matrix<double> outputErrors = point->getAnswer() - output;
    matrix<double> weight_hot = trans(weight_ho);
    matrix<double> hiddenErrors = prod(weight_hot, outputErrors);


    /////////adjusting weights with calculated errors/////////////

    matrix<double> gradient = output;
    matrix<double> hiddenGradient = hidden;


    for(int i = 0; i < gradient.size1(); i++){
        for(int j = 0; j < gradient.size2(); j++){
            gradient(i,j) = dsigmoidFunction(gradient(i,j));
            gradient(i,j) *= outputErrors(i,j);
        }
    }

    gradient *= lRate;
    matrix<double> hiddenT = trans(hidden);
    matrix<double> weight_ho_deltas = prod(gradient, hiddenT);
    //tuning weight matrix between hidden and output
    weight_ho += weight_ho_deltas;


    for(int i = 0; i < hiddenGradient.size1(); i++){
        for(int j = 0; j < hiddenGradient.size2(); j++){
            hiddenGradient(i,j) = dsigmoidFunction(hiddenGradient(i,j));
            hiddenGradient(i,j) *= hiddenErrors(i,j);
        }
    }

    hiddenGradient *= lRate;
    matrix<double> inputsT = trans(point->getInputs());
    matrix<double> weight_ih_deltas = prod(hiddenGradient,inputsT);
    //tuning weight matrix between input and hidden
    weight_ih += weight_ih_deltas;

    //tuning bias
    bias_o += gradient;
    bias_h += hiddenGradient;

}
