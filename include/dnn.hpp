#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP
#pragma once

#include <string>
#include <tuple>
#include <vector>
#include <cstddef>

#include <opencv2/opencv.hpp>

/**
  * @brief File contains declarations for functionality to run a deep neural network
  */

namespace vision_fsai {

    struct Detection {
        /**
          * @brief Detection (class) - basic struct to store detected object properties
          */
        int class_id ;

        cv::Rect bounding_box ;

        float confidence ;

        /**
          * @brief Detection (constructor) - initialises struct properties
          * @param const int - ID of class detected
          * @param const cv::Rect& - const reference to cv Rect containing bounding box start and ends of x and y respectively
          * @param const float - confidence that object of type ID was detected
          */
        inline Detection(const int, const ::cv::Rect&, const float) noexcept ;

    } ;

    /**
      * @brief insertion operator (overload) - prints content of detection struct
      * @param std::ostream& - reference to streamable object
      * @param const vision_fsai::Detection& - const reference to struct containing detected object properties
      * @return std::ostream& - (same) reference to streamable object
    */
    ::std::ostream& operator<<(::std::ostream&, const Detection&) ;

    class NeuralNetwork {
    /**
      * @brief NeuralNetwork (class) - class wrapped around to abstract opencv dnn
      */
        private:
            // general stuff
            ::cv::dnn::Net _net ; // opencv network object
            ::std::pair<cv::dnn::Backend, cv::dnn::Target> _backend ;
            ::std::vector<int> _output_ids ;
            ::std::vector<::std::string> _output_names ;
            ::cv::Size _input_size ;
            std::size_t _classes ;

            // input stuff
            ::cv::Mat _input ;
            ::cv::Size _frame_size ; // size of instance of input

            // output stuff
            ::std::vector<::cv::Mat> _outputs ;
            ::std::vector<Detection> _detections ;

            /**
              * @brief process_input - internal method which takes regular image, formats and processes them
              * @param const cv::Mat& - const reference to Mat storing image to process
              * @throws std::invalid_argument - thrown if image given contains nada
              */
            void process_input(const ::cv::Mat&) noexcept(false) ;

            /**
              * @brief process_output - internal method which deals with outputted result of the running of the neural network on an image
              * @param const float - number storing correlation threshold require to accept a finding as a detected object
              * @param const float - number storing threshold used for non maxima suppression of bounding box detections
              */
            void process_output(const float, const float) noexcept ;

            /**
              * @brief process_detection_network_output - internal method which deals with detection network output
              * @param const float - number storing correlation threshold require to accept a finding as a detected object
              * @return std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<cv::Rect>>> - returns lists of confidences for each detected type of object and their respecvtive bounding boxes
              */
            ::std::tuple<::std::vector<::std::vector<float>>, ::std::vector<::std::vector<::cv::Rect>>> process_detection_network_output(const float) noexcept ;

            /**
              * @brief process_region_network_output - internal method which deals with outputted result of the running of the neural network on an image
              * @param const float - number storing correlation threshold require to accept a finding as a detected object
              */
            ::std::tuple<::std::vector<::std::vector<float>>, ::std::vector<::std::vector<::cv::Rect>>> process_region_network_output(const float) noexcept ;

        public:
            /**
              * @brief NeuralNetwork (constructor) - initialises neural network with required calibration and setup information
              * @param const std::string& - const reference to string containing path to configuration file
              * @param const std::string& - const reference to string containing path to weights file
              * @param const cv::dnn::Backend - enum tag to specify specific computation backend. defaults to CUDA
              * @param const cv::dnn::Target - enum tag to specify which device the networks computes on. defaults to CUDA
              */
            NeuralNetwork(const ::std::string&, const ::std::string&, const ::cv::dnn::Backend=cv::dnn::DNN_BACKEND_CUDA, const ::cv::dnn::Target=cv::dnn::DNN_TARGET_CUDA) noexcept ;

            /**
              * @brief process - method which takes regular image, formats and processes them
              * @param const cv::Mat& - const reference to Mat storing image to process
              * @param const float - number storing correlation threshold require to accept a finding as a detected object
              * @param const float - number storing threshold used for non maxima suppression of bounding box detections
              */
            void process(const ::cv::Mat&, const float=0.5f, const float=0.4f) noexcept(false) ;

            /**
              * @brief dectecions - getter method which returns list of bounding box images detected by dnn
              * @return std::vector<vision_fsai::Detection> - list of detected cone objects
              */
            ::std::vector<Detection> detections() const noexcept ;

            /**
              * @brief classes - quasi getter method which returns class output information (ids it spits name, names it correlaates to)
              * @return std::vector<int> - tuple storing both list of class ids and class names
              */
            ::std::vector<int> classes() const noexcept ;

            // below are defaulted and deleted methods
            NeuralNetwork(const NeuralNetwork&) noexcept = default ;
            NeuralNetwork& operator=(const NeuralNetwork&) noexcept = default ;
            NeuralNetwork(NeuralNetwork&&) noexcept = default ;
            NeuralNetwork& operator=(NeuralNetwork&&) noexcept = default ;
            ~NeuralNetwork() noexcept = default ;
            NeuralNetwork() = delete ; // initialisation of both cnn and object happen in defined constructor.
                                       // empty constructor will cause program to crash and burn

    } ;

} ; // vision_fsai

#endif // NEURAL_NETWORK_HPP
