#ifndef GUN_VISION_H
#define GUN_VISION_H

#include "FrameFilter.h"
#include "BackgroundSubtractor.h"
#include "ArmConvolver.h"
#include "GunConvolver.h"
#include "NeuralNetwork.h"


class GunVision {

public:

   GunVision(unsigned int width, unsigned int height, bool shouldFlip);
   ~GunVision();

   // primary function of gun vision
   // receives frame from camera to search for raised gun
   // outputs subtracted frame for gui
   // offsets arm and gun pointers with locations if detected
   void detect(uint8_t* frame, uint8_t* subtractionHost,
               unsigned int& armX, unsigned int& armY,
               unsigned int& gunX, unsigned int& gunY);


   // uses frame to prepare background for future subtraction
   // initializes gpu variables
   void firstFrame(uint8_t* frame);


   // indicates gpu programs successfully compiled and gpu buffers allocated
   bool isSetup_;

private:

   // applies blur filter to new frame
   FrameFilter frameFilter_;

   // detects moving elements of frame versus static background
   BackgroundSubtractor backgroundSubtractor_;

   // uses template to search frame for a raised arm
   ArmConvolver armConvolver_;

   // uses template to search frame for a gun
   GunConvolver gunConvolver_;

   // uses area of original frame where gun template matched as input to neural network
   NeuralNetwork neuralNetwork_;
};


#endif //GUN_VISION_H
