#include "conv2d.h"
#include <ctime>
#include <iostream>
#include <benchmark/benchmark.h>

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


int main()
{
    double start = clock();

    
    Mat a(3, 3, 1);
    Mat b(2, 2, 1);
    Mat c;
    for(int i = 0; i < 10000; ++i)
        c = conv_3x3(a, b, 1);

    double end = clock();

    std::cout << c << std::endl;
    printf("Multiplication Time clock(): %fs\n", (double)(end - start)); ///  10 /CLOCKS_PER_SEC
}

