#include <iostream>
#include "NeuralNetwork.h"
#include "Point.h"

int main() {
    std::cout << "my life suck" << std::endl;
    matrix<double> m1(2,1);
    m1(0,0) = 1;
    m1(1,0) = 0;

    matrix<double> a1(2,1);
    a1(0,0) = 1;
    a1(1,0) = 0;

    point_ptr point(new Point(m1,a1));
    neuralNetwork_ptr network(new NeuralNetwork(2,2,2));

    network->train(point);
    return 0;
}