#include <GLUT/glut.h>
#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "GunVision.h"


cv::VideoCapture cap_;
uint8_t* subtractionHost_;
GunVision* gunVision_;
unsigned int width_ = 0;
unsigned int height_ = 0;


// open gl function to resize window - avoid doing so!
void reshape(int width, int height) {
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, width, height, 0);
    glMatrixMode(GL_MODELVIEW);
}


// open gl function for idle gui
void idle(void) {
    glutPostRedisplay();
}


// open gl function handling main display loop
void display(void) {

   // skip first several frames to obtain more stable data
   static const unsigned int FIRST_FRAME = 100;
   static unsigned int count = 0;
   count++;
   if(count < FIRST_FRAME) return;

   glClearColor(0, 0, 0, 0);
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

   // capture camera frame with open cv
   cv::Mat frame;
   cap_ >> frame;
   uint8_t* pixelPtr = (uint8_t*)frame.data;

   // use one frame to initialize GunVision_2.0
   if(count == FIRST_FRAME) {
      gunVision_->firstFrame(pixelPtr);
      return;
   }

   // will be side effected by GunVision_2.0 if detected
   unsigned int armX = 0;
   unsigned int armY = 0;
   unsigned int gunX = 0;
   unsigned int gunY = 0;

   // call primary detection function of GunVision_2.0
   gunVision_->detect(pixelPtr, subtractionHost_, armX, armY, gunX, gunY);

   // active elements of background subtracted frame are drawn in white
   glColor3d(1.0f, 1.0f, 1.0f);
   glBegin(GL_POINTS);

   unsigned int bufferIndex = 0;
   for(unsigned int h = 0; h<height_; h++) {
      for(unsigned int w = 0; w<width_; w++) {

         if(subtractionHost_[bufferIndex] == 1) glVertex2i(w,h);
         bufferIndex++;
      }
   }

   // draw arm in green if detected
   if(armX != 0) {

      glColor3d(0.0f, 1.0f, 0.0f);
      for(unsigned int h = 0; h<60; h++) {
        for(unsigned int w = 0; w<290; w++) {
           glVertex2i(w+armX, h+armY+15);
        }
      }
   }

   // draw gun in red if detected
   if(gunX != 0) {

      glColor3d(1.0f, 0.0f, 0.0f);
      for(unsigned int h = 0; h<100; h++) {
        for(unsigned int w = 0; w<190; w++) {
           glVertex2i(w+gunX, h+gunY+35);
        }
      }
   }

   glEnd();
   glutSwapBuffers();
}


// captures a camera frame with open cv and extracts dimensions
// initializes GunVision_2.0 and allocates gui buffers using frame size
// creates GLUT window and starts display loop
int main(int argc, char **argv) {

   bool shouldFlip = false;
   if(argc > 1) shouldFlip = true;

   cap_.set(cv::CAP_PROP_CONVERT_RGB, 1);
   cap_.open(0);
   cv::Mat frame;
   cap_ >> frame;

   width_ = frame.cols;
   height_ = frame.rows;

   std::cout << "Initializing GunVision_2.0 - width: " << width_ << "   height: " << height_ << std::endl;
   gunVision_ = new GunVision(width_, height_, shouldFlip);

   if(gunVision_->isSetup_ == false) {

      std::cout << "GunVision_2.0 initialization failed... terminating" << std::endl;
      return 0;
   }

   subtractionHost_ = new uint8_t[width_ * height_];

   glutInit(&argc, argv);
   glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
   glutInitWindowSize(width_, height_);
   glutCreateWindow("GunVision_2.0");
   glutDisplayFunc(display);
   glutReshapeFunc(reshape);
   glutIdleFunc(idle);

   glutMainLoop();

   return 1;
}
