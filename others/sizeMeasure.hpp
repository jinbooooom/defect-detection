#ifndef SIZEMEASURE_HPP
#define SIZEMEASURE_HPP

inline double length(Vec2i p1, Vec2i p2);
inline double oLength(Vec2i p1, Vec2i p2);
void vertexRecognition(Mat, int, vector<Vec2i> &);
bool judgement(vector<Vec2i> &vertexSorted, int mode, int &modelNumber, const int edgeError = 7, const double ratioError = 0.2);

#endif