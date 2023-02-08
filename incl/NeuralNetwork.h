#ifndef NEURAL_NET_H
#define NEURAL_NET_H


#include "OpenClBuilder.h"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


class NeuralNetwork : public OpenClBuilder {

public:

   NeuralNetwork();
   virtual ~NeuralNetwork();

   virtual void setSourceCode();

   virtual bool compile();

   // run forward algorithm of pre-trained neural network
   float forward();

private:

   std::string loadInputSource_;
   cl_kernel loadInputKernel_;

   bool isModelLoaded_;
   float* transitionBuffer_;
   cv::Mat inputLayer_;
   cv::dnn::Net neuralNetwork_;
};






#endif //NEURAL_NET_H
