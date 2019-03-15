//
// Created by kamil on 3/10/19.
//

#include "Point.h"

Point::Point(double x, double y, int a) {
    inputs = matrix<double>(1,2);
    inputs(0,0) = x;
    inputs(0,1) = y;
    answer = a;
}

matrix<double> Point::getInputs() {
    return inputs;
}

int Point::getAnswer() {
    return answer;
}
