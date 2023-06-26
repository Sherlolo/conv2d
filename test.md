# conv
baseline: 93 - 98
# conv_avx
18 * 18
conv_3x3_avx: 84 - 86
conv avx kernel_pre: 80 - 81
add block: 63
add stream and steam: 68 - 71
add _mm256_cvtss_f32: 60 - 63
add omp: 59 - 61

# conv_img2col

conv_img2col: 226 - 233
feature add avx: 99 - 105
add segem: 45
