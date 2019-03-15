#include <iostream>
#include "NeuralNetwork.h"
#include "Point.h"

int main() {
    std::cout << "my life suck" << std::endl;
    point_ptr point(new Point(1,0,0));
    neuralNetwork_ptr network(new NeuralNetwork(2,2,1));
    matrix<double> p1 = point->getInputs();
    //network->feedForward(p1);
    network->train(point);
    return 0;
}