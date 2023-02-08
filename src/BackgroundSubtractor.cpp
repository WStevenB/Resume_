#include "BackgroundSubtractor.h"


BackgroundSubtractor::BackgroundSubtractor() {

   subtractSource_ = "";
   filterSource_ = "";

   subtractKernel_ = NULL;
   filterKernel_ = NULL;
}


BackgroundSubtractor::~BackgroundSubtractor() {

}


void BackgroundSubtractor::setSourceCode() {

   // totals red, green, and blue difference per one pixel
   // outputs 1 if above threshold (dynamic element in frame) or 0 if below (just part of background)
   // also tracks certainty value per pixel, replaces background value when certainty falls below a threshold
   subtractSource_ =
   "__kernel void back_subtract(__global unsigned char* background, __global int* certainties, __global unsigned char* newFrame, __global unsigned char* subtraction, __global unsigned char* drop_factor) {"\
       "size_t i = get_global_id(0);"\
        "int loc = i*3;"\
        "int diff = background[loc] - newFrame[loc];"\
        "int diff_portion = -1;"\
        "if(diff < 0) { diff = diff * -1; diff_portion = 1; }"\
        "int diff2 = background[loc+1] - newFrame[loc+1];"\
        "int diff2_portion = -1;"\
        "if(diff2 < 0) { diff2 = diff2 * -1; diff2_portion = 1; }"\
        "int diff3 = background[loc+2] - newFrame[loc+2];"\
        "int diff3_portion = -1;"\
        "if(diff3 < 0) { diff3 = diff3 * -1; diff3_portion = 1; }"\

        "int diffTotal = diff + diff2 + diff3;"\

        "if(diffTotal > 40) {"\
           "drop_factor[i] = drop_factor[i] + 5;"\
           "if(drop_factor[i] > 2000) drop_factor[i] = 2000;"\
        "}"\
        "else {"\
         "drop_factor[i] = 1;"\
         "background[loc] = background[loc] + diff_portion;"\
         "background[loc+1] = background[loc+1] + diff2_portion;"\
         "background[loc+2] = background[loc+2] + diff3_portion;"\
        "}"\

        "if(diffTotal > 95)      { certainties[i] = certainties[i] - (150 * drop_factor[i]); subtraction[i] = 1; }"\
        "else if(diffTotal > 85) { certainties[i] = certainties[i] - (80 * drop_factor[i]);  subtraction[i] = 1; }"\
        "else if(diffTotal > 75) { certainties[i] = certainties[i] - (40 * drop_factor[i]);  subtraction[i] = 1; }"\
        "else if(diffTotal > 65) { certainties[i] = certainties[i] - (15 * drop_factor[i]);  subtraction[i] = 1; }"\
        "else if(diffTotal > 55) { certainties[i] = certainties[i] - (8 * drop_factor[i]);   subtraction[i] = 1; }"\
        "else if(diffTotal > 48) { certainties[i] = certainties[i] - (5 * drop_factor[i]);   subtraction[i] = 1; }"\
        "else if(diffTotal > 40) { certainties[i] = certainties[i] - (2 * drop_factor[i]);   subtraction[i] = 1;  }"\

        "else if(diffTotal > 25) { certainties[i] = certainties[i] + (2 * drop_factor[i]);   subtraction[i] = 0; }"\
        "else if(diffTotal > 15) { certainties[i] = certainties[i] + (3 * drop_factor[i]);   subtraction[i] = 0; }"\
        "else                    { certainties[i] = certainties[i] + (5 * drop_factor[i]);   subtraction[i] = 0; }"\

        "if(certainties[i] < -4000000) {"\
         "certainties[i] = 100;"\
         "drop_factor[i] = 1;"\
         "background[loc] = newFrame[loc];"\
         "background[loc+1] = newFrame[loc+1];"\
         "background[loc+2] = newFrame[loc+2];"\
        "}"\
        "if(certainties[i] > 2140000000) {"\
         "certainties[i] = 2140000000;"\
        "}"\
   "}";


   // applies blur filter to subtracted frame
   filterSource_ =
   "__kernel void filter(__global unsigned char* subtraction, __global unsigned char* filter) {"\
      "size_t i = get_global_id(0);"\
      "unsigned int width = i%" + __WIDTH__ + ";"\
      "unsigned int height = i/" + __WIDTH__ + ";"\
      "if(width >= 3 && width <= " + __CLIPPED_WIDTH__ + " && height >= 3 && height <= " + __CLIPPED_HEIGHT__ + ") {"\
         "unsigned char total = 0;"\
         "for(int h = -3; h<=3; h++) {"\
            "for(int w = -3; w<=3; w++) {"\
               "int loc = ((height + h) * " + __WIDTH__ + ") + width + w;"\
               "total = total + subtraction[loc];"\
            "}"\
         "}"\
         "if(total > 30) filter[i] = 1;"\
         "else filter[i] = 0;"\
      "}"\
      "else filter[i] = 0;"\
   "}";
}


bool BackgroundSubtractor::compile() {

   cl_int error;

   const char* subtractSourcePtr = subtractSource_.c_str();
   cl_program subtractProgram = clCreateProgramWithSource(context_, 1, (const char **) &subtractSourcePtr, NULL, &error);
   if (clBuildProgram(subtractProgram, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS) {
      std::cout << "Error compiling subtract" << std::endl;
      return false;
   }
   subtractKernel_ = clCreateKernel(subtractProgram, "back_subtract", &error);

   const char* filterSourcePtr = filterSource_.c_str();
   cl_program filterProgram = clCreateProgramWithSource(context_, 1, (const char **) &filterSourcePtr, NULL, &error);
   if (clBuildProgram(filterProgram, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS) {
      std::cout << "Error compiling filter" << std::endl;
      return false;
   }
   filterKernel_ = clCreateKernel(filterProgram, "filter", &error);

   return true;
}


void BackgroundSubtractor::subtract(uint8_t* subtractionHost) {

   // perform background subtraction
   clSetKernelArg(subtractKernel_, 0, sizeof(cl_mem), &clBackground_);
   clSetKernelArg(subtractKernel_, 1, sizeof(cl_mem), &clCertainties_);
   clSetKernelArg(subtractKernel_, 2, sizeof(cl_mem), &clNewFrame_);
   clSetKernelArg(subtractKernel_, 3, sizeof(cl_mem), &clSubtracted_);
   clSetKernelArg(subtractKernel_, 4, sizeof(cl_mem), &clDropFactors_);

   clEnqueueNDRangeKernel(commandQueue_, subtractKernel_, 1, NULL, &pixelSize_, NULL, 0, NULL, NULL);
   clFinish(commandQueue_);


   // apply blur filter
   clSetKernelArg(filterKernel_, 0, sizeof(cl_mem), &clSubtracted_);
   clSetKernelArg(filterKernel_, 1, sizeof(cl_mem), &clSubtractedFiltered_);

   clEnqueueNDRangeKernel(commandQueue_, filterKernel_, 1, NULL, &pixelSize_, NULL, 0, NULL, NULL);
   clFinish(commandQueue_);


   // extract from gpu to display to user
   clEnqueueReadBuffer(commandQueue_, clSubtractedFiltered_, CL_TRUE, 0, sizeof(uint8_t)*pixelSize_, subtractionHost, 0, NULL, NULL);
}
