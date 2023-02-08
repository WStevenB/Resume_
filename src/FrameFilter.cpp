#include "FrameFilter.h"


FrameFilter::FrameFilter(bool shouldFlip) {

   shouldFlip_ = shouldFlip;
   firstFrameSource_ = "";
   flipFrameSource_ = "";
   filterFrameSource_ = "";
   firstKernel_ = NULL;
   flippedKernel_ = NULL;
   filterKernel_ = NULL;
}


FrameFilter::~FrameFilter() {

}

void FrameFilter::setSourceCode() {

   // copies new frame to background
   // initializes certainties and drop factors
   firstFrameSource_ =
   "__kernel void copy_first_frame(__global unsigned char* first_frame, __global unsigned char* background, __global int* certainties, __global unsigned char* drop) {"\
      "size_t i = get_global_id(0);"\
      "background[i] = first_frame[i];"\
      "if(i<" + __PIXEL_SIZE__ + ") {"\
         "certainties[i] = 100; drop[i] = 1;"\
      "}"\
   "}";


   // flips frame horizontally and stores in new buffer used instead during future processing
   flipFrameSource_ =
   "__kernel void flip_frame(__global unsigned char* raw_frame, __global unsigned char* flipped_frame) {"\
      "size_t i = get_global_id(0);"\
      "unsigned int width = i%" + __WIDTH__ + ";"\
      "unsigned int height = i/" + __WIDTH__ + ";"\
      "unsigned int flipped_width = " + __WIDTH__ + " - width - 1;"\
      "unsigned int buffer_location = ((height * " + __WIDTH__ + ") + width)*3;"\
      "unsigned int flipped_location = ((height * " + __WIDTH__ + ") + flipped_width)*3;"\
      "flipped_frame[flipped_location] = raw_frame[buffer_location];"\
      "flipped_frame[flipped_location+1] = raw_frame[buffer_location+1];"\
      "flipped_frame[flipped_location+2] = raw_frame[buffer_location+2];"\
   "}";


   // applies blur filter to raw frame
   filterFrameSource_ =
   "__kernel void filter_frame(__global unsigned char* raw_frame, __global unsigned char* filter_frame) {"\
      "size_t i = get_global_id(0);"\
      "unsigned int loc_f = i*3;"\
      "unsigned int width = i%" + __WIDTH__ + ";"\
      "unsigned int height = i/" + __WIDTH__ + ";"\
      "if(width >= 3 && width <= " + __CLIPPED_WIDTH__ + " && height >= 3 && height <= " + __CLIPPED_HEIGHT__ + ") {"\
         "unsigned int total_1 = 0;"\
         "unsigned int total_2 = 0;"\
         "unsigned int total_3 = 0;"\
         "for(int h = -2; h<=2; h++) {"\
            "for(int w = -2; w<=2; w++) {"\
               "unsigned int loc = (((height + h) * " + __WIDTH__ + ") + width + w) * 3;"\
               "total_1 = total_1 + (unsigned int) raw_frame[loc];"\
               "total_2 = total_2 + (unsigned int) raw_frame[loc+1];"\
               "total_3 = total_3 + (unsigned int) raw_frame[loc+2];"\
            "}"\
         "}"\
         "filter_frame[loc_f] = (unsigned char) (total_1 / 25);"\
         "filter_frame[loc_f+1] = (unsigned char) (total_2 / 25);"\
         "filter_frame[loc_f+2] = (unsigned char) (total_3 / 25);"\
      "}"\
      "else {"\
         "filter_frame[loc_f] = raw_frame[loc_f];"\
         "filter_frame[loc_f+1] = raw_frame[loc_f+1];"\
         "filter_frame[loc_f+2] = raw_frame[loc_f+2];"\
       "}"\
   "}";
}





bool FrameFilter::compile() {

   cl_int error;

   const char* firstFrameSourcePtr = firstFrameSource_.c_str();
   cl_program firstProgram = clCreateProgramWithSource(context_, 1, (const char **) &firstFrameSourcePtr, NULL, &error);
   if (clBuildProgram(firstProgram, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS) {
      std::cout << "Error compiling first frame" << std::endl;
      return false;
   }
   firstKernel_ = clCreateKernel(firstProgram, "copy_first_frame", &error);

   const char* flipFrameSourcePtr = flipFrameSource_.c_str();
   cl_program flippedProgram = clCreateProgramWithSource(context_, 1, (const char **) &flipFrameSourcePtr, NULL, &error);
   if (clBuildProgram(flippedProgram, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS) {
      std::cout << "Error compiling flip frame" << std::endl;
      return false;
   }
   flippedKernel_ = clCreateKernel(flippedProgram, "flip_frame", &error);

   const char* filterFrameSourcePtr = filterFrameSource_.c_str();
   cl_program filterProgram = clCreateProgramWithSource(context_, 1, (const char **) &filterFrameSourcePtr, NULL, &error);
   if (clBuildProgram(filterProgram, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS) {
      std::cout << "Error compiling filter frame" << std::endl;
      return false;
   }
   filterKernel_ = clCreateKernel(filterProgram, "filter_frame", &error);

   return true;
}


// prepare frame for background subtraction by applying blur filter
// optionally flip frame horizontally first
void FrameFilter::filter(uint8_t* frame) {

   // load frame from camera into gpu
   clEnqueueWriteBuffer(commandQueue_, clRawFrame_, CL_TRUE, 0, sizeof(uint8_t) * bufferSize_, frame, 0, NULL, NULL);

   if(shouldFlip_) {

      // flip frame horizontally
      clSetKernelArg(flippedKernel_, 0, sizeof(cl_mem), &clRawFrame_);
      clSetKernelArg(flippedKernel_, 1, sizeof(cl_mem), &clFlippedFrame_);

      clEnqueueNDRangeKernel(commandQueue_, flippedKernel_, 1, NULL, &pixelSize_, NULL, 0, NULL, NULL);
      clFinish(commandQueue_);

      // set flipped frame as argument to blur filter
      clSetKernelArg(filterKernel_, 0, sizeof(cl_mem), &clFlippedFrame_);
   }
   else {
      // no flipping desired, set raw frame as argument to blur filter
      clSetKernelArg(filterKernel_, 0, sizeof(cl_mem), &clRawFrame_);
   }

   clSetKernelArg(filterKernel_, 1, sizeof(cl_mem), &clNewFrame_);

   // run blur filter
   clEnqueueNDRangeKernel(commandQueue_, filterKernel_, 1, NULL, &pixelSize_, NULL, 0, NULL, NULL);
   clFinish(commandQueue_);
}


// runs blur filter and sets background values
// initializes gpu variables
void FrameFilter::firstFrame(uint8_t* frame) {

   // apply blur filter
   filter(frame);

   // store first frame in background and initialize certanties and drop factors in gpu
   clSetKernelArg(firstKernel_, 0, sizeof(cl_mem), &clNewFrame_);
   clSetKernelArg(firstKernel_, 1, sizeof(cl_mem), &clBackground_);
   clSetKernelArg(firstKernel_, 2, sizeof(cl_mem), &clCertainties_);
   clSetKernelArg(firstKernel_, 3, sizeof(cl_mem), &clDropFactors_);

   clEnqueueNDRangeKernel(commandQueue_, firstKernel_, 1, NULL, &bufferSize_, NULL, 0, NULL, NULL);
   clFinish(commandQueue_);
}
