#include "conv2d.h"
#include <ctime>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <benchmark/benchmark.h>
#include <omp.h>

//void test_conv2d(benchmark::State& bm)
//{
//    //double start = clock();
//
//    Mat a(3,3,1);
//    Mat b(2,2,1);
//    Mat c = conv_3x3(a,b,1);
//    //double end = clock();
//
//    /*std::cout << c << std::endl;
//    printf("Multiplication Time clock(): %fs\n", (double)(end - start) / 10 /CLOCKS_PER_SEC);*/
//}
//BENCHMARK(test_conv2d);
//
//BENCHMARK_MAIN();

typedef Mat (*FuncPtr)(const Mat& input, const Mat& kernel, const int stride);

void init_mat(Mat& m, const float X)
{
    for (int i = 0; i < m.w * m.h; ++i)
        m.data[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / X));
}

void test_conv(const Mat& a, const Mat& b, FuncPtr fun)
{
    Mat c;
    //auto t0 = std::chrono::steady_clock::now();
    double start = clock();
    double start1 = omp_get_wtime();
    for (int i = 0; i < 50000; ++i)
        c = (*fun)(a, b, 1);
    double end = clock();
    double end1 = omp_get_wtime();
    //auto t1 = std::chrono::steady_clock::now();
    //std::cout << c << std::endl;
    printf("Multiplication Time clock(): %f clock\n", (double)(end - start)); 
    printf("Multiplication Time omp_clock(): %f clock\n", (double)(end1 - start1));
    //printf("Multiplication Time clock(): %ld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count()); 
}

int main()
{
    const float X = 10;
    std::srand(0);

    Mat a(18, 18);
    Mat b(3, 3);
    init_mat(a, X);
    init_mat(b, X);

    test_conv(a, b, conv_3x3);
    test_conv(a, b, conv_3x3_img2col);

    return 0;
}

