#include "Mat.h"
#include <vector>

Mat conv_3x3(const Mat& input, const Mat& kernel, const int stride);
Mat conv_winograd(const Mat& input, const Mat& kernel, const int stride);
Mat conv_3x3_img2col(const Mat& input, const Mat& kernel, const int stride);
//Mat conv_3x3_img2col_pack(const Mat& input, const Mat& kernel, const int stride);
