#ifndef ARM_CONVOLVER_H
#define ARM_CONVOLVER_H


#include "OpenClBuilder.h"


class ArmConvolver : public OpenClBuilder {

public:

   ArmConvolver();
   virtual ~ArmConvolver();

   virtual void setSourceCode();

   virtual bool compile();

   // searches subtracted frame for raised arm
   void convolve(int* armTotalsHost);


private:

   std::string subtotalsSource_;
   std::string totalsSource_;

   cl_kernel subtotalsKernel_;
   cl_kernel totalsKernel_;
};







#endif //ARM_CONVOLVER_H
