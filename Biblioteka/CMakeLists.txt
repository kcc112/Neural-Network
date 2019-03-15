cmake_minimum_required(VERSION 3.4)
project(Biblioteka)

SET(SOURCE_FILES include/NeuralNetwork.h src/NeuralNetwork.cpp include/Point.h src/Point.cpp)

#utwórz bibliotekę typu STATIC, SHARED albo MODULE ze wskazanych źródeł
add_library(Biblioteka ${SOURCE_FILES})

target_include_directories(Biblioteka PUBLIC include)