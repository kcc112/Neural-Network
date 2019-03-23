//
// Created by kamil on 3/10/19.
//

#include "Point.h"

//answer is true score
Point::Point(matrix<double> input, matrix<int> answer) {
    this->inputs = input;
    this->answer = answer;
}

matrix<double> Point::getInputs() {
    return inputs;
}

matrix<int> Point::getAnswer() {
    return answer;
}
