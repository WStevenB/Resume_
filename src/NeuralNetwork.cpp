#include "NeuralNetwork.h"


NeuralNetwork::NeuralNetwork() {

   loadInputSource_ = "";
   loadInputKernel_ = NULL;

   std::string model = "model/frozen_graph.pb";
   neuralNetwork_ = cv::dnn::readNetFromTensorflow(model);

   if(neuralNetwork_.empty()) {
      isModelLoaded_ = false;
      std::cout << "Loading neural network from model failed"<< std::endl;
   }
   else isModelLoaded_ = true;

   inputLayer_ = cv::Mat(1, NETWORK_INPUT_SIZE, CV_32FC1);

   transitionBuffer_ = new float[NETWORK_INPUT_SIZE]();
}


NeuralNetwork::~NeuralNetwork() {

}


void NeuralNetwork::setSourceCode() {

   // prepare input layer of neural network using area of original frame where gun was detected
   // reduce the size of this area from 190x135 (size of gun template) to 63x45 using blur filter
   // recenter pixel values from 0:255 to -5:5 and then perform sigmoid function
   // bias of 0.65 is used on last value of input layer
   loadInputSource_ =
   "__kernel void net_start(__global unsigned char* raw_frame, __global unsigned char* background,"\
                           "__global float* net_input_layer, __global int* gunTotals) {"\
      "size_t i = get_global_id(0);"\
      "if(i == 2835) { net_input_layer[8505] = 0.65f; }"\
      "else {"\
         "unsigned int netLoc = i*3;"\
         "unsigned int x = ((i%63) * 3) + 1;"\
         "unsigned int y = ((i/63) * 3) + 1;"\
         "float totalRed = 0.0f;"\
         "float totalBlue = 0.0f;"\
         "float totalGreen = 0.0f;"\

         "for(int xx = -1; xx<2; xx++) {"\
            "for(int yy = -1; yy<2; yy++) {"\

               "int xxx = x+xx+gunTotals[1];"\
               "int yyy = y+yy+gunTotals[2];"\
               "int subtractionLoc = (yyy*" + __WIDTH__ + ") + xxx;"\

               "if(background[subtractionLoc] == 0) {"\
                  "totalRed += 255.0f;"\
                  "totalBlue += 255.0f;"\
                  "totalGreen += 255.0f;"\
               "}"\
               "else {"\
                  "int rawLoc = subtractionLoc * 3;"\
                  "totalRed += (float)raw_frame[rawLoc];"\
                  "totalBlue += (float)raw_frame[rawLoc+1];"\
                  "totalGreen += (float)raw_frame[rawLoc+2];"\
               "}"\
            "}"\
         "}"\
         "net_input_layer[netLoc] = 1.0f / ( 1.0f + exp( -1.0f * ((totalRed/9.0f) * (10.0f/255.0f) - 5.0f)));"\
         "net_input_layer[netLoc+1] = 1.0f / ( 1.0f + exp( -1.0f * ((totalBlue/9.0f) * (10.0f/255.0f) - 5.0f)));"\
         "net_input_layer[netLoc+2] = 1.0f / ( 1.0f + exp( -1.0f * ((totalGreen/9.0f) * (10.0f/255.0f) - 5.0f)));"\
      "}"\
   "}";
}


bool NeuralNetwork::compile() {

   cl_int error;

   const char* loadInputPtr = loadInputSource_.c_str();
   cl_program inputProgram = clCreateProgramWithSource(context_, 1, (const char **) &loadInputPtr, NULL, &error);
   if (clBuildProgram(inputProgram, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS) {
      std::cout << "Error compiling load neural net input" << std::endl;
      return false;
   }
   loadInputKernel_ = clCreateKernel(inputProgram, "net_start", &error);

   return true;
}


// run forward algorithm of pre-trained neural network
float NeuralNetwork::forward() {

   if(isModelLoaded_ == false) {
      std::cout << "Neural network model not loaded" << std::endl;
      return 0;
   }

   // prepare input layer
   clSetKernelArg(loadInputKernel_, 0, sizeof(cl_mem), &clNewFrame_);
   clSetKernelArg(loadInputKernel_, 1, sizeof(cl_mem), &clSubtractedFiltered_);
   clSetKernelArg(loadInputKernel_, 2, sizeof(cl_mem), &clNetworkInput_);
   clSetKernelArg(loadInputKernel_, 3, sizeof(cl_mem), &clGunTotals_);

   clEnqueueNDRangeKernel(commandQueue_, loadInputKernel_, 1, NULL, &NETWORK_INPUT_PIXELS, NULL, 0, NULL, NULL);
   clFinish(commandQueue_);

   // extract neural network input layer from gpu and copy to open cv
   clEnqueueReadBuffer(commandQueue_, clNetworkInput_, CL_TRUE, 0,
                       sizeof(float)*NETWORK_INPUT_SIZE, transitionBuffer_, 0, NULL, NULL);

   for(unsigned int i = 0; i<NETWORK_INPUT_SIZE; i++) inputLayer_.at<float>(0,i) = transitionBuffer_[i];
   neuralNetwork_.setInput(inputLayer_);

   // run Tensorflow neural network and return output
   cv::Mat output = neuralNetwork_.forward();
   return output.at<float>(0,0);
}




