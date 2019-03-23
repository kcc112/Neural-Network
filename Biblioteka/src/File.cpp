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

void File::writeToFile(std::string filename, std::vector<double> input) {

    std::string line;
    std::fstream file;

    file.open(filename, std::ios::out | std::ios::trunc);
    if(file.good())
    {
        double sum = 0;
        for( int i = 0; i < input.size();) {
            for ( int j = 0; j < 10 + i; j ++) {
                sum += input[j];
            }
            i += 10;
            double output = sum / i;
            sum = 0;
            file << i << " " << output << std::endl;
        }
        file.close();
    }
}
