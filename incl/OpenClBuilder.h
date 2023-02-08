#ifndef OPENCL_BUILDER_H
#define OPENCL_BUILDER_H


#include <string>
#include <iostream>
#include <OpenCL/opencl.h>


class OpenClBuilder {

public:

   OpenClBuilder();
   virtual ~OpenClBuilder();

   // sets open cl using dynamic values of frame size
   virtual void setSourceCode() = 0;

   // compiles open cl programs
   virtual bool compile() = 0;

   // set dimensions, performs open cl setup tasks, and creates gpu buffers
   static bool initialize(unsigned long width, unsigned long height);


protected:

   // open cl variables to administer gpu processing
   // shared instance between all classes running open cl programs
   static cl_context context_;
   static cl_command_queue commandQueue_;

   // shared gpu data buffers between all class running open cl programs
   static cl_mem clRawFrame_;
   static cl_mem clNewFrame_;
   static cl_mem clFlippedFrame_;
   static cl_mem clBackground_;
   static cl_mem clCertainties_;
   static cl_mem clDropFactors_;
   static cl_mem clSubtracted_;
   static cl_mem clSubtractedFiltered_;
   static cl_mem clArmSubtotals_;
   static cl_mem clArmTotals_;
   static cl_mem clGunTemplate_;
   static cl_mem clGunSubtotals_;
   static cl_mem clGunTotals_;
   static cl_mem clNetworkInput_;

   // dimensions related to frame size and others
   // used to create gpu buffers and spawn specific amounts of programs
   static unsigned long width_;
   static unsigned long height_;
   static unsigned long pixelSize_;
   static unsigned long bufferSize_;
   static unsigned long clippedWidth_;
   static unsigned long clippedHeight_;

   static const unsigned long GUN_SIZE;
   static const unsigned long GUN_WIDTH;
   static const unsigned long NETWORK_INPUT_SIZE;
   static const unsigned long NETWORK_INPUT_PIXELS;

   // string version of dimensions used in open cl source code
   static std::string __WIDTH__;
   static std::string __HEIGHT__;
   static std::string __PIXEL_SIZE__;
   static std::string __BUFFER_SIZE__;
   static std::string __CLIPPED_WIDTH__;
   static std::string __CLIPPED_HEIGHT__;

};






#endif //OPENCL_BUILDER_H
