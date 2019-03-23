//
// Created by kamil on 3/14/19.
//

#ifndef NEURALNETWORK_NEURALNETWORK_H
#define NEURALNETWORK_NEURALNETWORK_H

#include <boost/numeric/ublas/matrix.hpp>
#include <memory>

class Point;

typedef std::shared_ptr<Point> point_ptr;

using namespace boost::numeric::ublas;

class NeuralNetwork {

    double lRate;
    matrix<double> weight_ih;
    matrix<double> weight_ho;
    matrix<double > bias_h;
    matrix<double> bias_o;

public:
    NeuralNetwork(unsigned int inputNodes, unsigned int hiddenNodes, unsigned int outputNodes);
    ~NeuralNetwork() = default;
    matrix<double> feedForward(matrix<double> input);
    matrix<double> calculateHidden(matrix<double> input);
    matrix<double> calculateOutput(matrix<double> input);
    double sigmoidFunction(double x);
    double dsigmoidFunction(double x);
    void randomizeMatrix(matrix<double> & input, int minV,int maxV);
    void train(point_ptr point);
};

typedef std::shared_ptr<NeuralNetwork> neuralNetwork_ptr;

#endif //NEURALNETWORK_NEURALNETWORK_H
