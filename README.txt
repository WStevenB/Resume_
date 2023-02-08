GunVision_3.0 is configured to run on an M1 Macbook Pro.
It shouldn't be too difficult to modify the Makefile for Linux or Windows.
You'll need to install OpenGL and OpenCV. Then provide paths to those libraries.
More effort would be required to port to Android or iOS, but should be possible.

GunVision_3.0 attempts to detect a raised handgun using background subtraction, template matching for an arm,
template matching for a gun, and a Tensorflow neural network.
Imagine it being used at a bank or gas station near the counter.
If a criminal points a gun at the cashier or teller, then GunVision_3.0 could notify the police.

Run the binary from the root folder of the repo so it knows the path to the Tensorflow model.
./bin/GunVision_3.0

Default detection direction is gun pointing to the right side of the picture frame.
Add an argument to invert the camera frame to detect guns aimed the opposite direction.
The idea is to launch GunVision_3.0 according to which side a bank teller or cashier is standing relative to the camera.
./bin/GunVision_3.0 -r

Ideal detection distance is around seven feet from camera to gun, and camera 3-4 feet off the ground.
