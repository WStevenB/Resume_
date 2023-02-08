#ifndef BACKGROUND_SUBTRACTOR_H
#define BACKGROUND_SUBTRACTOR_H


#include "OpenClBuilder.h"


class BackgroundSubtractor : public OpenClBuilder {

public:

   BackgroundSubtractor();
   virtual ~BackgroundSubtractor();

   virtual void setSourceCode();

   virtual bool compile();

   // filters out unchanged elements of new frame using previous frames
   void subtract(uint8_t* subtractionHost);


private:

   std::string subtractSource_;
   std::string filterSource_;

   cl_kernel subtractKernel_;
   cl_kernel filterKernel_;
};






#endif //BACKGROUND_SUBTRACTOR_H
