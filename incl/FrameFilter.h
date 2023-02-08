#ifndef FRAME_FILTER_H
#define FRAME_FILTER_H


#include "OpenClBuilder.h"


class FrameFilter : public OpenClBuilder {

public:

   FrameFilter(bool shouldFlip);
   virtual ~FrameFilter();

   virtual void setSourceCode();

   virtual bool compile();

   // prepare frame for background subtraction by applying blur filter
   // optionally flip frame horizontally first
   void filter(uint8_t* frame);

   // runs blur filter and sets background values
   // initializes gpu variables
   void firstFrame(uint8_t* frame);


private:

   // default false - gun expected to be pointed to the right when facing camera
   // set true when command line argument indicates gun is pointed to the left
   bool shouldFlip_;

   std::string firstFrameSource_;
   std::string flipFrameSource_;
   std::string filterFrameSource_;

   cl_kernel firstKernel_;
   cl_kernel flippedKernel_;
   cl_kernel filterKernel_;

};





#endif //FRAME_FILTER_H
