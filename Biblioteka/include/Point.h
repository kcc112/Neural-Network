//
// Created by kamil on 3/10/19.
//

#ifndef PERCEPTRON_POINT_H
#define PERCEPTRON_POINT_H

#include <memory>
#include <vector>
#include <boost/numeric/ublas/matrix.hpp>

using namespace boost::numeric::ublas;

class Point {

    matrix<double> inputs;
    int answer;

public:
    Point() = default;
    Point(double x, double y, int a);
    ~Point() = default;
    matrix<double> getInputs();
    int getAnswer();
};

typedef std::shared_ptr<Point> point_ptr;

#endif //PERCEPTRON_POINT_H
