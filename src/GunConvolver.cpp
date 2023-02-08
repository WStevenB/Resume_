#include "GunConvolver.h"


GunConvolver::GunConvolver() {

   subtotalsSource_ = "";
   totalsSource_ = "";
   subtotalsKernel_ = NULL;
   totalsKernel_ = NULL;
}


GunConvolver::~GunConvolver() {

}


void GunConvolver::setSourceCode() {

   // calculates subtotals per pixel and per column of gun template
   // only searches above and to the left of arm
   // also avoids extreme edges of frame
   subtotalsSource_ =
   "__kernel void gun_subtotals(__global unsigned char* frame, __global int* totals, __global int* gun, __global int* arm) {"\
      "size_t i = get_global_id(0);"\
      "totals[i] = 0;"\
      "unsigned int gunCol = i/" + __PIXEL_SIZE__ + ";"\
      "unsigned int gunOffset = gunCol * 135;"\
      "unsigned int remainder = i%" + __PIXEL_SIZE__ + ";"\
      "unsigned int width = remainder%" + __WIDTH__ + ";"\
      "unsigned int height = remainder/" + __WIDTH__ + ";"\
      "unsigned int loc = (((height) * " + __WIDTH__ + ") + width);"\
      "int total = 0;"\
      "if(width >= 5 && height >= 100 && width < arm[1] + 100 && height < arm[2] + 50) {"\
         "for(unsigned int h = gunOffset; h<135+gunOffset; h++) {"\
            "total = total + (frame[loc] * gun[h]);"\
            "loc = loc + " + __WIDTH__ + ";"\
         "}"\
         "totals[i] = total;"\
      "}"\
   "}";


   // adds subtotals for gun template score per pixel
   totalsSource_ =
   "__kernel void gun_totals(__global int* subtotals, __global int* totals, __global int* arm) {"\
      "size_t i = get_global_id(0);"\
      "totals[i] = 0;"\
      "unsigned int width = i%" + __WIDTH__ + ";"\
      "unsigned int height = i/" + __WIDTH__ + ";"\
      "unsigned int loc = (((height) * " + __WIDTH__ + ") + width);"\
      "int total = 0;"\
      "if(width >= 5 && height >= 100 && width < arm[1] + 100 && height < arm[2] + 50) {"\
         "for(unsigned int w = 0; w<190; w++) {"\
            "total = total + (subtotals[loc]);"\
            "loc = loc + " + __PIXEL_SIZE__ + " + 1;"\
         "}"\
         "totals[i] = total;"\
         "if(total > 6500 && total > totals[0]) {"\
            "totals[0] = total;"\
            "totals[1] = width;"\
            "totals[2] = height;"\
         "}"\
      "}"\
   "}";
}


bool GunConvolver::compile() {

   cl_int error;

   const char* subtotalsPtr = subtotalsSource_.c_str();
   cl_program subtotalsProgram = clCreateProgramWithSource(context_, 1, (const char **) &subtotalsPtr, NULL, &error);
   if (clBuildProgram(subtotalsProgram, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS) {
      std::cout << "Error compiling gun subtotals" << std::endl;
      return false;
   }
   subtotalsKernel_ = clCreateKernel(subtotalsProgram, "gun_subtotals", &error);

   const char* totalsPtr = totalsSource_.c_str();
   cl_program totalsProgram = clCreateProgramWithSource(context_, 1, (const char **) &totalsPtr, NULL, &error);
   if (clBuildProgram(totalsProgram, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS) {
      std::cout << "Error compiling gun totals" << std::endl;
      return false;
   }
   totalsKernel_ = clCreateKernel(totalsProgram, "gun_totals", &error);

   return true;
}


// search subtracted frame for gun template
void GunConvolver::convolve(int* gunTotalsHost) {

   // calculate subtotal for each pixel and each column of gun template
   clSetKernelArg(subtotalsKernel_, 0, sizeof(cl_mem), &clSubtractedFiltered_);
   clSetKernelArg(subtotalsKernel_, 1, sizeof(cl_mem), &clGunSubtotals_);
   clSetKernelArg(subtotalsKernel_, 2, sizeof(cl_mem), &clGunTemplate_);
   clSetKernelArg(subtotalsKernel_, 3, sizeof(cl_mem), &clArmTotals_);

   size_t global = pixelSize_ * GUN_WIDTH;
   clEnqueueNDRangeKernel(commandQueue_, subtotalsKernel_, 1, NULL, &global, NULL, 0, NULL, NULL);
   clFinish(commandQueue_);


   // add up subtotals for gun template score per pixel
   // stored as pixel is upper left corner of template
   clSetKernelArg(totalsKernel_, 0, sizeof(cl_mem), &clGunSubtotals_);
   clSetKernelArg(totalsKernel_, 1, sizeof(cl_mem), &clGunTotals_);
   clSetKernelArg(totalsKernel_, 2, sizeof(cl_mem), &clArmTotals_);

   clEnqueueNDRangeKernel(commandQueue_, totalsKernel_, 1, NULL, &pixelSize_, NULL, 0, NULL, NULL);
   clFinish(commandQueue_);


   // extract results from gpu - [template score, x location, y location]
   clEnqueueReadBuffer(commandQueue_, clGunTotals_, CL_TRUE, 0, sizeof(int)*3, gunTotalsHost, 0, NULL, NULL);
}
