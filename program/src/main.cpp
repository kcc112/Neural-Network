#include <iostream>
//#include <boost/numeric/ublas/matrix.hpp>
#include "NeuralNetwork.h"

int main() {
    std::cout << "Hello, World!" << std::endl;
    neuralNetwork_ptr network(new NeuralNetwork(2,2,1));
    matrix<double> p1(2,1);
    p1(0,0) = 1;
    p1(1,0) = 2;
    network->feedForward(p1);
    return 0;
}