#include <iostream>
#include <boost/numeric/ublas/io.hpp>
#include "NeuralNetwork.h"
#include "Point.h"
#include "File.h"

int main() {

    file_ptr file(new File());

    std::vector<point_ptr> pointArray;
    std::vector<int> trainData = file->readFromFile("test.txt");
    matrix<double> m(4,1);
    matrix<double> a(4,1);

    for(int j = 0; j < trainData.size(); j+=4) {
        for (int i = 0; i < 4; i++) {
            m(i,0) = trainData[i+j];
            a(i,0) = trainData[i+j];
        }
        point_ptr p(new Point(m, a));
        pointArray.push_back(p);
    }

    neuralNetwork_ptr network(new NeuralNetwork(4,3,4));

    for (int j = 0; j < 10000; ++j) {
        for (int i = 0; i < 4; ++i) {
            network->train(pointArray[i]);
        }
    }

    file->writeToFile("Averag-error3.txt", network->getDifPow());

    for (int i = 0; i < 4; i++) {
        std::cout << network->feedForward(pointArray[i]->getInputs()) << std::endl;
    }

    /*matrix<double> m1(2,1);
    m1(0,0) = 1;
    m1(1,0) = 0;
    matrix<double> a1(1,1);
    a1(0,0) = 1;

    matrix<double> m2(2,1);
    m2(0,0) = 0;
    m2(1,0) = 1;
    matrix<double> a2(1,1);
    a2(0,0) = 1;

    matrix<double> m3(2,1);
    m3(0,0) = 0;
    m3(1,0) = 0;
    matrix<double> a3(1,1);
    a3(0,0) = 0;

    matrix<double> m4(2,1);
    m4(0,0) = 1;
    m4(1,0) = 1;
    matrix<double> a4(1,1);
    a4(0,0) = 0;

    point_ptr point1(new Point(m1,a1));
    point_ptr point2(new Point(m2,a2));
    point_ptr point3(new Point(m3,a3));
    point_ptr point4(new Point(m4,a4));


    point_ptr points[4];

    points[0] = point1;
    points[1] = point2;
    points[2] = point3;
    points[3] = point4;

    neuralNetwork_ptr network(new NeuralNetwork(2,2,1));

    for (int j = 0; j < 10000; ++j) {
        for (int i = 0; i < 4; ++i) {
            network->train(points[i]);
        }
    }


    std::cout << network->feedForward(m1) << std::endl;
    std::cout << network->feedForward(m2) << std::endl;
    std::cout << network->feedForward(m3) << std::endl;
    std::cout << network->feedForward(m4) << std::endl;*/

    return 0;
}
