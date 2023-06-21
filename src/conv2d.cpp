#include "conv2d.h"

Mat conv_3x3(const Mat& input, const Mat& kernel, const int stride){
    int outw = (input.w - kernel.w)/stride + 1; //(inw- kernelw + 2*p)/stride + 1
    int outh = (input.h - kernel.h)/stride + 1;
    int kernel_size = kernel.w*kernel.h;
    Mat output(outw, outh);

    std::vector<int> in_bias(kernel_size);
    for(int k = 0; k < kernel_size; ++k)
    {
        in_bias[k] = k%kernel.w + (k/kernel.w)*input.w;
    }


    for(int i = 0; i < outh; ++i)
    {
        for(int j = 0; j < outw; ++j)
        {
            float* in_data = &input(i*stride, j*stride);
            float sum = 0;
            for(int k = 0; k < kernel_size; ++k)
            {
                sum += in_data[in_bias[k] ] * kernel.data[k];
            }
            output(i, j) = sum;
        }
    }

    return output;
}

Mat conv_3x3_img2col(const Mat& input, const Mat& kernel, const int stride)
{
    int outw = (input.w - kernel.w)/stride + 1; //(inw- kernelw + 2*p)/stride + 1
    int outh = (input.h - kernel.h)/stride + 1;
    int kernel_size = kernel.w*kernel.h;
    Mat output(outw, outh);

    std::vector<int> in_bias(kernel_size);
    for(int k = 0; k < kernel_size; ++k)
    {
        in_bias[k] = k%kernel.w + (k/kernel.w)*input.w;
    }
    //input2col
    //Mat in_mat(outw*outh, kernel_size); 
    Mat in_mat(kernel_size, outw*outh); 
    for(int i = 0; i < outh; ++i)
    {
        for(int j = 0; j < outw; ++j)
        {
            float* in_data = &input(i*stride, j*stride);
            for(int k = 0; k < kernel_size; ++k)
            {
                in_mat(i*outw+j, k) = in_data[in_bias[k] ];
            }
        }
    }

    //kernel2col
    float* in_kernel = kernel.data;

    //in_mat * kernel
    for(int i = 0; i < in_mat.h; ++i)
    {
        for(int j = 0; j < in_mat.w; ++j)
        {
            float sum = 0;
            sum += in_mat(i,j) * in_kernel[j];
            output.data[i] = sum;
        }
    }

    return output;
}
