#include <string>
#include <iostream>

#include <opencv2/opencv.hpp>

#include "dnn.hpp"

int main(const int argc, const char** argv)
{
    cv::Mat cone_img = cv::imread(std::string{argv[3]}) ;

    vision_fsai::NeuralNetwork dnn{std::string{argv[1]}, std::string{argv[2]}, cv::dnn::getAvailableBackends()[0].first, cv::dnn::getAvailableBackends()[0].second} ;
    dnn.process(cone_img) ;

    const auto detections = dnn.detections() ;
    if(!detections.empty())
    {
        for(const auto& i : detections) std::cout << i << '\n' ;
        std::cout.flush() ;
    }
    else {
        std::cout << "No objects detected!" << std::endl ;
    }

    /* E(nd) O(f) P(rograms) */
    return 0 ;
}
