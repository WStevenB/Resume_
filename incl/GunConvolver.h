#ifndef GUN_CONVOLVER_H
#define GUN_CONVOLVER_H


#include "OpenClBuilder.h"


class GunConvolver : public OpenClBuilder {

public:

   GunConvolver();
   virtual ~GunConvolver();

   virtual void setSourceCode();

   virtual bool compile();

   // search subtracted frame for gun template
   void convolve(int* gunTotalsHost);


private:

   std::string subtotalsSource_;
   std::string totalsSource_;

   cl_kernel subtotalsKernel_;
   cl_kernel totalsKernel_;
};







#endif //GUN_CONVOLVER_H
