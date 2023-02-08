#include "ArmConvolver.h"


ArmConvolver::ArmConvolver() {

   subtotalsSource_ = "";
   totalsSource_ = "";
   subtotalsKernel_ = NULL;
   totalsKernel_ = NULL;
}


ArmConvolver::~ArmConvolver() {

}


void ArmConvolver::setSourceCode() {

   // calculates column value of arm template at each pixel
   // template is negative zone (not arm), neutral buffer, positive zone (arm), another neutral buffer, and negative zone
   // optimized for no sleeves at 5-10 feet
   subtotalsSource_ =
   "__kernel void arm_subtotals(__global unsigned char* frame, __global int* totals) {"\
      "size_t i = get_global_id(0);"\
      "totals[i] = 0;"\
      "unsigned int width = i%" + __WIDTH__ + ";"\
      "unsigned int height = i/" + __WIDTH__ + ";"\
      "unsigned int loc = (((height) * " + __WIDTH__ + ") + width);"\
      "unsigned int clippedWidth = " + __WIDTH__ + "-50;"\
      "unsigned int clippedHeight = " + __HEIGHT__ + "-100;"\
      "int total = 0;"\
      "if(width >= 50 && width <= clippedWidth && height >= 150 && height <= clippedHeight) {"\
         "for(unsigned int h = 0; h<120; h++) {"\
            "int factor = 1;"\
            "if(h<15 || h>105) factor = -5;"\
            "else if(h<30 || h>90) factor = 0;"\
            "total = total + (frame[loc] * factor);"\
            "loc = loc + " + __WIDTH__ + ";"\
         "}"\
         "totals[i] = total;"\
      "}"\
   "}";


   // iterates horizontally for width of gun template summing column subtotals
   // calculates arm template score per pixel
   // if greater than threshold and current max, stores value and locations
   totalsSource_ =
   "__kernel void arm_totals(__global int* subtotals, __global int* totals) {"\
      "size_t i = get_global_id(0);"\
      "totals[i] = 0;"\
      "int width = i%" + __WIDTH__ + ";"\
      "int height = i/" + __WIDTH__ + ";"\
      "int loc = (((height) * " + __WIDTH__ + ") + width);"\
      "unsigned int clippedWidth = " + __WIDTH__ + "-290;"\
      "unsigned int clippedHeight = " + __HEIGHT__ + "-100;"\
      "int total = 0;"\
      "if(width >= 50 && width <= clippedWidth && height >= 150 && height <= clippedHeight) {"\
         "for(unsigned int w = 0; w<290; w++) {"\
            "total = total + (subtotals[loc]);"\
            "loc = loc + 1;"\
         "}"\
         "totals[i] = total;"\
         "if(total > 9000 && total > totals[0]) {"\
            "totals[0] = total;"\
            "totals[1] = width;"\
            "totals[2] = height;"\
         "}"\
      "}"\
   "}";
}


bool ArmConvolver::compile() {

   cl_int error;

   const char* subtotalsPtr = subtotalsSource_.c_str();
   cl_program subtotalsProgram = clCreateProgramWithSource(context_, 1, (const char **) &subtotalsPtr, NULL, &error);
   if (clBuildProgram(subtotalsProgram, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS) {
      std::cout << "Error compiling arm subtotals" << std::endl;
      return false;
   }
   subtotalsKernel_ = clCreateKernel(subtotalsProgram, "arm_subtotals", &error);

   const char* totalsPtr = totalsSource_.c_str();
   cl_program totalsProgram = clCreateProgramWithSource(context_, 1, (const char **) &totalsPtr, NULL, &error);
   if (clBuildProgram(totalsProgram, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS) {
      std::cout << "Error compiling arm totals" << std::endl;
      return false;
   }
   totalsKernel_ = clCreateKernel(totalsProgram, "arm_totals", &error);

   return true;
}


// searches subtracted frame for raised arm
void ArmConvolver::convolve(int* armTotalsHost) {

   // calculate column subtotals
   clSetKernelArg(subtotalsKernel_, 0, sizeof(cl_mem), &clSubtractedFiltered_);
   clSetKernelArg(subtotalsKernel_, 1, sizeof(cl_mem), &clArmSubtotals_);

   clEnqueueNDRangeKernel(commandQueue_, subtotalsKernel_, 1, NULL, &pixelSize_, NULL, 0, NULL, NULL);
   clFinish(commandQueue_);


   // add up columns for arm template score
   clSetKernelArg(totalsKernel_, 0, sizeof(cl_mem), &clArmSubtotals_);
   clSetKernelArg(totalsKernel_, 1, sizeof(cl_mem), &clArmTotals_);

   clEnqueueNDRangeKernel(commandQueue_, totalsKernel_, 1, NULL, &pixelSize_, NULL, 0, NULL, NULL);
   clFinish(commandQueue_);


   // extract results from gpu - [template score, x location, y location]
   clEnqueueReadBuffer(commandQueue_, clArmTotals_, CL_TRUE, 0, sizeof(int)*3, armTotalsHost, 0, NULL, NULL);
}



