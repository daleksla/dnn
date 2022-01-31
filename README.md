# dnn
## README

Simple wrapper of [OpenCV](https://opencv.org)'s deep neural network implementation.

To process an image requires three steps:
1. Initialise NeuralNetwork object, supplying configuration file, weights file, computation backend and a computation target (latter two default to CUDA if no specified)
2. Call `process` method with a valid image
3. Call `detections` method to then retrieve a list of struct containing what object and where it was detected and the confidence of said detection.

***

See LICENSE.md for licensing details
