#include "conv2d.h"
#include <vector>
#include <immintrin.h>
#include <omp.h>

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

Mat conv_3x3_avx0(const Mat& input, const Mat& kernel, const int stride)
{
    int outw = (input.w - kernel.w)/stride + 1; 
    int outh = (input.h - kernel.h)/stride + 1;
    int kernel_size = kernel.w*kernel.h;
    Mat output(outw, outh);
    std::vector<int> in_bias(kernel_size);
    for(int k = 0; k < kernel_size; ++k)
    {
        in_bias[k] = k%kernel.w + (k/kernel.w)*input.w;
    }

    __m256 kernel_256 = _mm256_load_ps((kernel.data+1));
    //#pragma omp parallel for num_threads(6)
    for(int i = 0; i < outh; ++i)
    {
        //_mm_prefetch((const char*)(&input(i*stride+1, 0)), _MM_HINT_T0);
        //_mm_prefetch((const char*)(&input(i*stride+2, 0)), _MM_HINT_T0);
        for(int j = 0; j < outw; j++)
        {
            //_mm_prefetch((const char*)(&input(i*stride+kernel.w, j*stride)), _MM_HINT_T0);
       
            float* in_data = &input(i*stride, j*stride);
            float sum = in_data[in_bias[0]]*kernel.data[0];
            __m256 sum_256 = _mm256_setr_ps(in_data[in_bias[1]], in_data[in_bias[2]], in_data[in_bias[3]], in_data[in_bias[4]],
            in_data[in_bias[5]], in_data[in_bias[6]], in_data[in_bias[7]], in_data[in_bias[8]]);
            sum_256 = _mm256_mul_ps(sum_256, kernel_256);
            sum_256 = _mm256_hadd_ps(sum_256, sum_256);
            sum_256 = _mm256_hadd_ps(sum_256, sum_256);

            sum += sum_256.m256_f32[0];            // add res
            sum += sum_256.m256_f32[4];            // add res
            
            output(i,j) = sum;
        }
    }

    return output;
}

Mat conv_3x3_avx(const Mat& input, const Mat& kernel, const int stride)
{
    int outw = (input.w - kernel.w)/stride + 1; 
    int outh = (input.h - kernel.h)/stride + 1;
    int kernel_size = kernel.w*kernel.h;
    Mat output(outw, outh);
    std::vector<int> in_bias(kernel_size);
    for(int k = 0; k < kernel_size; ++k)
    {
        in_bias[k] = k%kernel.w + (k/kernel.w)*input.w;
    }

    __m256 kernel_256 = _mm256_load_ps((kernel.data+1));
    #pragma omp parallel for num_threads(8) //collapse(2)
    for(int i = 0; i < outh; ++i)
    {
        //_mm_prefetch((const char*)(&input(i*stride+1, 0)), _MM_HINT_T0);
        //_mm_prefetch((const char*)(&input(i*stride+2, 0)), _MM_HINT_T0);
        for(int j = 0; j < outw; j+=8)
        {
            //_mm_prefetch((const char*)(&input(i*stride, j*stride)), _MM_HINT_T0);
         
            __m256 sum_arr;
            for(int k = 0; k < 8; ++k)
            {
                float* in_data = &input(i*stride, j*stride+k);
                float sum = in_data[in_bias[0]]*kernel.data[0];
                __m256 sum_256 = _mm256_setr_ps(in_data[in_bias[1]], in_data[in_bias[2]], in_data[in_bias[3]], in_data[in_bias[4]],
                in_data[in_bias[5]], in_data[in_bias[6]], in_data[in_bias[7]], in_data[in_bias[8]]);
                sum_256 = _mm256_mul_ps(sum_256, kernel_256);
                sum_256 = _mm256_hadd_ps(sum_256, sum_256);
                sum_256 = _mm256_hadd_ps(sum_256, sum_256);

                sum += sum_256.m256_f32[0];            // add res
                sum += sum_256.m256_f32[4];            // add res
                sum_arr.m256_f32[k] = sum;
            }

            _mm256_storeu_ps(&output(i, j), sum_arr);
            //_mm256_stream_ps(&output(i, j), sum_arr);
        }
    }

    return output;
}

Mat conv_3x3_avx2(const Mat& input, const Mat& kernel, const int stride)
{
    int outw = (input.w - kernel.w)/stride + 1; 
    int outh = (input.h - kernel.h)/stride + 1;
    int kernel_size = kernel.w*kernel.h;
    Mat output(outw, outh);
    std::vector<int> in_bias(kernel_size);
    for(int k = 0; k < kernel_size; ++k)
    {
        in_bias[k] = k%kernel.w + (k/kernel.w)*input.w;
    }

    __m256 kernel_256 = _mm256_load_ps((kernel.data+1));
    //#pragma omp parallel for num_threads(6)
    for(int i = 0; i < outh; ++i)
    {
        //_mm_prefetch((const char*)(&input(i*stride+1, 0)), _MM_HINT_T0);
        //_mm_prefetch((const char*)(&input(i*stride+2, 0)), _MM_HINT_T0);
        for(int j = 0; j < outw; j+=8)
        {
            //_mm_prefetch((const char*)(&input(i*stride, j*stride)), _MM_HINT_T0);
         
            __m256 sum_arr;
            __m256 input_arr[8];
            for(int k = 0; k < 8; ++k)
            {
                float* in_data = &input(i*stride, j*stride+k);
                sum_arr.m256_f32[k] = in_data[in_bias[0]]*kernel.data[0];
                input_arr[k] = _mm256_setr_ps(in_data[in_bias[1]], in_data[in_bias[2]], in_data[in_bias[3]], in_data[in_bias[4]],
                in_data[in_bias[5]], in_data[in_bias[6]], in_data[in_bias[7]], in_data[in_bias[8]]);
            }

            for(int k = 0; k < 8; ++k)
            {
                
                input_arr[k] = _mm256_mul_ps(input_arr[k], kernel_256);
                input_arr[k] = _mm256_hadd_ps(input_arr[k], input_arr[k]);
                input_arr[k] = _mm256_hadd_ps(input_arr[k], input_arr[k]);

                sum_arr.m256_f32[k] += input_arr[k].m256_f32[0];            // add res
                sum_arr.m256_f32[k] += input_arr[k].m256_f32[4];  
            }


            _mm256_storeu_ps(&output(i, j), sum_arr);
            //_mm256_stream_ps(&output(i, j), sum_arr);
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
