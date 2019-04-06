//
// Created by kamil on 3/23/19.
//

#ifndef PERCEPTRON_FILE_H
#define PERCEPTRON_FILE_H

#include <string>
#include <memory>
#include <vector>

class File {
public:
    File() = default;
    ~File() = default;
    std::vector<int> readFromFile(std::string filename);
    void writeToFile(std::string filename, std::vector<double> input, unsigned int outputNodes, unsigned int trainingSize);

};

typedef std::shared_ptr<File> file_ptr;

#endif //PERCEPTRON_FILE_H
