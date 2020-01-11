#ifndef ROTATEIMAGE_HPP
#define ROTATEIMAGE_HPP

IplImage* rotateImage(IplImage* src, double angle, bool clockwise);
double getAngle(Mat &img);

#endif