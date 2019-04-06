//
// Created by kamil on 3/23/19.
//

#include "File.h"
#include <fstream>
#include <iostream>

std::vector<int> File::readFromFile(std::string filename) {
    int x;
    std::vector<int> output;
    std::ifstream inFile;

    inFile.open(filename);
    if (!inFile) {
        std::cout << "Unable to open file";
        exit(1); // terminate with error
    }

    while (inFile >> x) {
        output.push_back(x);
    }

    inFile.close();

    return output;
}

void File::writeToFile(std::string filename, std::vector<double> input, unsigned int outputNodes, unsigned int trainingSize) {

    std::string line;
    std::fstream file;

    file.open(filename, std::ios::out | std::ios::trunc);
    if(file.good())
    {
        double sum = 0;
        int era = 1;
        for(int i = 0; i < input.size(); i += trainingSize) {
            for(int j = 0; j < trainingSize; j++) {
                sum += input[i + j];
            }
            file << era << " " << sum / (2 * outputNodes * trainingSize) << std::endl;
            era ++;
            sum = 0;
        }
        file.close();
    }
}