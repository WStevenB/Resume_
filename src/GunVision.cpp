#include "GunVision.h"

#include <iostream>


GunVision::GunVision(unsigned int width, unsigned int height, bool shouldFlip) :
   isSetup_(true),
   frameFilter_(shouldFlip),
   backgroundSubtractor_(),
   armConvolver_(),
   gunConvolver_(),
   neuralNetwork_() {

   // allocates open cl buffers and sets variables
   if(OpenClBuilder::initialize(width, height) == false) {
      isSetup_ = false;
      return;
   }

   // uses frame dimensions to prepare open cl code
   frameFilter_.setSourceCode();
   backgroundSubtractor_.setSourceCode();
   armConvolver_.setSourceCode();
   gunConvolver_.setSourceCode();
   neuralNetwork_.setSourceCode();

   // compiles open cl code into runnable kernels
   if( frameFilter_.compile()
    && backgroundSubtractor_.compile()
    && armConvolver_.compile()
    && gunConvolver_.compile()
    && neuralNetwork_.compile() ) {

      std::cout << "Open CL programs successfully compiled" << std::endl;
   }
   else isSetup_ = false;

}


GunVision::~GunVision() {

}


// primary function of gun vision
// receives frame from camera to search for raised gun
// outputs subtracted frame for gui
// offsets arm and gun pointers with locations if detected
void GunVision::detect(uint8_t* frame, uint8_t* subtractionHost,
                       unsigned int& armX, unsigned int& armY,
                       unsigned int& gunX, unsigned int& gunY) {

   frameFilter_.filter(frame);
   backgroundSubtractor_.subtract(subtractionHost);

   static int armTotalsHost[3];
   armTotalsHost[0] = 0;
   armConvolver_.convolve(armTotalsHost);

   if(armTotalsHost[0] > 0) {

      armX = static_cast<unsigned int>(armTotalsHost[1]);
      armY = static_cast<unsigned int>(armTotalsHost[2]);

      static int gunTotalsHost[3];
      gunTotalsHost[0] = 0;
      gunConvolver_.convolve(gunTotalsHost);

      if(gunTotalsHost[0] > 0) {

         std::cout << "Gun template matched with score " << gunTotalsHost[0] << std::endl;

         float output = neuralNetwork_.forward();

         std::cout << "Neural network result is " << output << std::endl;

         if(output > 0.85) {

            std::cout << "Gun detected!!!" << std::endl << std::endl;
            gunX = static_cast<unsigned int>(gunTotalsHost[1]);
            gunY = static_cast<unsigned int>(gunTotalsHost[2]);
         }
         else {
            std::cout << "Not a gun..." << std::endl << std::endl;
         }
      }
   }
}


// uses frame to prepare background for future subtraction
// initializes gpu variables
void GunVision::firstFrame(uint8_t* frame) {

   frameFilter_.firstFrame(frame);
}
