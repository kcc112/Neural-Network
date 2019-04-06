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

    unsigned int inputNodes;
    unsigned int hiddenNodes;
    unsigned int outputNodes;
    unsigned int hiddenSize;
    std::vector<double> difPow;
    int biaSS;
    double lRate;
    double momentum;
    std::vector<matrix<double>> weights;
    std::vector<matrix<double>> biasWeights;
    std::vector<matrix<double>> momentumWeights;

public:
    NeuralNetwork(unsigned int inputNodes, unsigned int hiddenNodes, unsigned int outputNodes, unsigned int hiddenSize, unsigned int period, unsigned int trainingSize, double momentum, double lRate, int biaS);
    ~NeuralNetwork() = default;
    matrix<double> feedForwardNew(const matrix<double> & input);
    matrix<double> calculateHiddenNew(const matrix<double> & input, int i);
    matrix<double> calculateOutputNew(const matrix<double> & input);
    double sigmoidFunction(double x);
    double dsigmoidFunction(double x);
    void randomizeMatrix(matrix<double> & input, double minV,double maxV);
    void cost(const matrix<double> & input);
    void trainNew(const point_ptr & point);
    std::vector<double> getDifPow();
};

typedef std::shared_ptr<NeuralNetwork> neuralNetwork_ptr;

#endif //NEURALNETWORK_NEURALNETWORK_H