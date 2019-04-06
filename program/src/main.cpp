#include <iostream>
#include <boost/numeric/ublas/io.hpp>
#include <string>
#include <chrono>
#include "NeuralNetwork.h"
#include "Point.h"
#include "File.h"

int main() {

    file_ptr file(new File());
    std::string fileName;
    int networkInput;
    int networkOutput;
    int hiddenNeurons;
    int hiddenSize;
    int period;
    int trainingSize;
    int bias;
    double momentum;
    double lRate;


    std::cout << "Podaj nazwe pliku z danymi do nauki" << std::endl;
    std::cin >> fileName;
    std::cout << "Podaj ilosc wejsc do sieci" << std::endl;
    std::cin >> networkInput;
    std::cout << "Podaj ilosc wyjsc z sieci" << std::endl;
    std::cin >> networkOutput;
    std::cout << "Podaj ilosc neuronow w warstwie ukrytej" << std::endl;
    std::cin >> hiddenNeurons;
    std::cout << "Podaj ilosc warstw ukrytych" << std::endl;
    std::cin >> hiddenSize;
    std::cout << "Podaj ilosc epok" << std::endl;
    std::cin >> period;
    std::cout << "Podaj rozmiar zbioru treningowego" << std::endl;
    std::cin >> trainingSize;
    std::cout << "Podaj momentum" << std::endl;
    std::cin >> momentum;
    std::cout << "Podaj learning rate" << std::endl;
    std::cin >> lRate;
    std::cout << "Z biasem 1 bez biasu 2" << std::endl;
    std::cin >> bias;

    std::vector<int> trainData = file->readFromFile(fileName);
    std::vector<point_ptr> pointArray;
    matrix<double> m(networkInput,1);
    matrix<double> a(networkInput,1);

    for(int j = 0; j < trainData.size(); j += networkInput * 2) {
        int k = 0;
        for (int i = 0; i < networkInput; i++) {
            m(i,0) = trainData[k + j];
            a(i,0) = trainData[k + j + 1];
            k += 2;
        }
        point_ptr p(new Point(m, a));
        pointArray.push_back(p);
    }

    auto start = std::chrono::steady_clock::now();
    neuralNetwork_ptr network(new NeuralNetwork(networkInput, hiddenNeurons, networkOutput, hiddenSize, period, trainingSize, momentum, lRate, bias));

    for (int j = 0; j < period; j++) {
        for (int i = 0; i < networkInput; i++) {
            network->trainNew(pointArray[i]);
        }
    }
    auto end = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end -start).count() << std::endl;
    std::cout << std::chrono::duration_cast<std::chrono::seconds>(end -start).count() << std::endl;
    file->writeToFile("Averag-error.txt", network->getDifPow(), networkOutput, trainingSize);
    std::cout << network->getDifPow().size() << std::endl;

    std::cout << std::endl;
    for (int i = 0; i < networkInput; i++) {
        std::cout << network->feedForwardNew(pointArray[i]->getInputs()) << std::endl;
    }

    return 0;
}
