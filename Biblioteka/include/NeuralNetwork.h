//
// Created by kamil on 3/14/19.
//

#ifndef NEURALNETWORK_NEURALNETWORK_H
#define NEURALNETWORK_NEURALNETWORK_H

#include <boost/numeric/ublas/matrix.hpp>
#include <memory>

using namespace boost::numeric::ublas;

class NeuralNetwork {

    unsigned int inputNodes;
    unsigned int hiddenNodes;
    unsigned int outputNodes;
    matrix<double> weight_ih;
    matrix<double> weight_ho;
    matrix<double > bias_h;
    matrix<double> bias_o;

public:
    NeuralNetwork(unsigned int inputNodes, unsigned int hiddenNodes, unsigned int outputNodes);
    ~NeuralNetwork() = default;
    int feedForward(matrix<double> input);

};

typedef std::shared_ptr<NeuralNetwork> neuralNetwork_ptr;

#endif //NEURALNETWORK_NEURALNETWORK_H
