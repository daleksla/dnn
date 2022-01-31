#include <string>
#include <vector>
#include <filesystem>
#include <tuple>
#include <stdexcept>
#include <cstddef>
#include <regex>
#include <fstream>
#include <iostream>

#include <opencv2/opencv.hpp>

#include "dnn.hpp"

/**
  * @brief File contains definitions for functionality to run a deep neural network
  */

// struct Detection related functionality

inline vision_fsai::Detection::Detection(const int i_class_id, const cv::Rect& i_rect, const float i_conf) noexcept : class_id(i_class_id), bounding_box(i_rect), confidence(i_conf) {} ;

std::ostream& vision_fsai::operator<<(std::ostream& os, const vision_fsai::Detection& detection)
{
    os << detection.class_id << " @ " << detection.bounding_box << " (confidence of detection: " << detection.confidence << ')' ;
    return os ;
}

// private methods Neural Network

void vision_fsai::NeuralNetwork::process_input(const cv::Mat& img) noexcept(false)
{
    if(img.empty())
    {
        const std::string msg = "Image data invalid / empty and cannot be processed" ;
        throw std::invalid_argument(msg) ;
    }
    this->_frame_size = img.size() ;

    cv::Mat blob ; cv::dnn::blobFromImage(img, blob, 1.f / 255.f, this->_input_size, cv::Scalar(), true, false );
    this->_net.setInput(blob) ;
}

void vision_fsai::NeuralNetwork::process_output(const float conf_threshold, const float nms_threshold) noexcept
{
    static std::vector<int> outLayers = this->_net.getUnconnectedOutLayers();
    static std::string outLayerType = this->_net.getLayer(outLayers[0])->type;

    if(outLayerType != "Region" && outLayerType != "DetectionOutput")
    {
        std::cerr << "Invalid outlayer type" << std::endl ;
        std::abort() ;
    }

    static const bool region = outLayerType == "Region";
    const auto [confidences, boxes] = (region ? this->process_region_network_output(conf_threshold) : this->process_detection_network_output(conf_threshold)) ;

    this->_detections.clear();

    if( outLayers.size() > 1 || ( region && this->_backend.first != cv::dnn::Backend::DNN_BACKEND_OPENCV ) )
    {
        for(std::size_t c = 0 ; c < this->_classes ; ++c)
        {
            std::vector<int> nmsIndices ; cv::dnn::NMSBoxes(boxes[c], confidences[c], conf_threshold, nms_threshold, nmsIndices) ;
            for(const int& idx : nmsIndices)
            {
                this->_detections.push_back( vision_fsai::Detection(c, boxes[c][idx], confidences[c][idx]) );
            }
        }
    }
    else {
        for(std::size_t c = 0 ; c < this->_classes ; ++c)
        {
            for(std::size_t i = 0 ; i < confidences[c].size() ; ++i)
            {
                this->_detections.push_back( vision_fsai::Detection(c, boxes[c][i], confidences[c][i]) ) ;
            }
        }
    }
}

std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<cv::Rect>>> vision_fsai::NeuralNetwork::process_detection_network_output(const float confThreshold) noexcept
{
    std::vector<std::vector<float>> confidences(this->_classes) ;
    std::vector<std::vector<cv::Rect>> boxes(this->_classes) ;

    // Network produces output blob with a shape 1x1xNx7 where N is a number of
    // detections and an every detection is a vector of values
    // [batchId, classId, confidence, left, top, right, bottom]
    for(std::size_t k = 0; k < this->_outputs.size(); k++)
    {
        float* data = reinterpret_cast<float*>(this->_outputs[k].data) ;
        for (std::size_t i = 0; i < this->_outputs[k].total(); i += 7)
        {
            float confidence = data[i + 2];
            if (confidence > confThreshold)
            {
                int left   = static_cast<int>( data[i + 3] );
                int top    = static_cast<int>( data[i + 4] );
                int right  = static_cast<int>( data[i + 5] );
                int bottom = static_cast<int>( data[i + 6] );
                int width  = right - left + 1;
                int height = bottom - top + 1;
                if (width <= 2 || height <= 2)
                {
                        left   = static_cast<int>( data[i + 3] * this->_frame_size.width );
                        top    = static_cast<int>( data[i + 4] * this->_frame_size.height );
                        right  = static_cast<int>( data[i + 5] * this->_frame_size.width );
                        bottom = static_cast<int>( data[i + 6] * this->_frame_size.height );
                        width  = right - left + 1;
                        height = bottom - top + 1;
                }

                const std::size_t classId = (int)(data[i+1]-1);
                confidences[classId].push_back( confidence );
                boxes[classId].push_back( cv::Rect(left, top, width, height) );
            }
        }
    }

    return std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<cv::Rect>>>{
        confidences, boxes
    } ;
}

std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<cv::Rect>>> vision_fsai::NeuralNetwork::process_region_network_output(const float confThreshold) noexcept
{
    std::vector<std::vector<float>> confidences(this->_classes) ;
    std::vector<std::vector<cv::Rect>> boxes(this->_classes) ;

    for (std::size_t i = 0; i < this->_outputs.size(); ++i)
    {
        // Network produces output blob with a shape NxC where N is a number of
        // detected objects and C is a number of classes + 4 where the first 4
        // numbers are [center_x, center_y, width, height]
        const float* data = reinterpret_cast<float*>(_outputs[i].data) ;
        for (int j = 0; j < this->_outputs[i].rows; ++j, data += this->_outputs[i].cols)
        {
            cv::Mat scores = _outputs[i].row(j).colRange(5, _outputs[i].cols);
            cv::Point classIdPoint;
            double confidence;
            cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if( confidence >= confThreshold )
            {
                const int centerX = static_cast<int>( data[0] * this->_frame_size.width );
                const int centerY = static_cast<int>( data[1] * this->_frame_size.height );
                const int width   = static_cast<int>( data[2] * this->_frame_size.width );
                const int height  = static_cast<int>( data[3] * this->_frame_size.height );
                const int left = centerX - width / 2;
                const int top = centerY - height / 2;

                confidences[ classIdPoint.x ].push_back( confidence );
                boxes[ classIdPoint.x ].push_back( cv::Rect( left, top, width, height ) );
            }
        }
    }

    return std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<cv::Rect>>>{
        confidences, boxes
    } ;
}

// public methods Neural Network

vision_fsai::NeuralNetwork::NeuralNetwork(const std::string& config_file, const std::string& weights_file, const cv::dnn::Backend network_backend, const cv::dnn::Target computation_target) noexcept
{
    auto verify_existing_file = [](const std::string& filename) -> bool {
        std::filesystem::path fp(filename) ;
        return std::filesystem::exists(fp) ;
    } ;

    if(!verify_existing_file(config_file) || !verify_existing_file(weights_file))
    {
        const std::string msg("YOLO files don't exist where specified") ;
        std::cerr << msg << std::endl ;
        std::abort() ;
    }

    auto read_width_height_from_cfg = [](const std::string& filename) -> cv::Size {
        const std::regex reg("(width|height)=([0-9]{1,})") ;

        cv::Size result { 0, 0 };

        std::ifstream ifs( filename );
        std::string line;
        while(std::getline(ifs, line) && (!result.width || !result.height))
        {
            std::smatch match;
            if( std::regex_match( line, match, reg ) )
                ( match[1] == "width" ? result.width : result.height ) = std::stoi( match[2] );
        }

        return result;
    } ;

    this->_input_size = read_width_height_from_cfg(config_file) ;
    if(this->_input_size.width <= 0 || this->_input_size.height <= 0)
    {
        std::cerr << "Invalid image dimensions read from provided config file" << std::endl ;
        std::abort() ;
    }

    const auto backends = cv::dnn::getAvailableBackends() ;
    this->_backend = std::pair<cv::dnn::Backend, cv::dnn::Target>{network_backend, computation_target} ;
    if( std::find( backends.begin(), backends.end(), _backend ) == backends.end())
    {
        std::cerr << "Invalid computation combination" << std::endl ;
        std::abort() ;
    }

    this->_net = cv::dnn::readNetFromDarknet(config_file, weights_file) ;

    auto get_output_names = [](const cv::dnn::Net& net) -> std::vector<std::string> {
        std::vector<int> outLayers = net.getUnconnectedOutLayers() ; //! Get the indices of the output layers, i.e. the layers with unconnected outputs

        std::vector<std::string> layersNames = net.getLayerNames() ; //! get the names of all the layers in the network

        //! Get the names of the output layers in names
        std::vector<std::string> names(outLayers.size()) ;
        for(std::size_t i = 0; i < outLayers.size(); ++i)
        {
            names[i] = layersNames[outLayers[i] - 1] ;
        }

        return names ;
    } ;

    this->_classes = 3 ;

    this->_output_names = get_output_names(this->_net) ;

    this->_net.setPreferableBackend(this->_backend.first) ;
    this->_net.setPreferableTarget(this->_backend.second) ;

    if(this->_net.empty())
    {
        const std::string msg = "Error creating neural network" ;
        std::abort() ;
    }
}

void vision_fsai::NeuralNetwork::process(const cv::Mat& img, const float conf_threshold, const float nms_threshold) noexcept(false)
{
    this->process_input(img) ;
    this->_net.forward(this->_outputs, this->_output_names) ;
    this->process_output(conf_threshold, nms_threshold) ;
}

std::vector<vision_fsai::Detection> vision_fsai::NeuralNetwork::detections() const noexcept
{
    return this->_detections ;
}

std::vector<int> vision_fsai::NeuralNetwork::classes() const noexcept
{
    return this->_output_ids ;
}
