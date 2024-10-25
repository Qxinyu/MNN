#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define GLOBAL_SIZE_3_DIMS __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,
#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                       \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                     \
    }

#define MNN_DATA_FORMAT_NCHW 0
#define MNN_DATA_FORMAT_NHWC 1
#define MNN_DATA_FORMAT_NC4HW4 2
#define MNN_DATA_FORMAT_C4NHW4 3
__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void gl_to_cl(GLOBAL_SIZE_3_DIMS
                                    __read_only image2d_t input_ptr,
                                    __private const int4 shape, // N C H W
                                    #ifdef USE_IMAGE
                                    __write_only image2d_t output_ptr
                                    #else
                                    __global OUTPUT_TYPE *output_ptr
                                    #endif
) {

    int wh  = get_global_id(0);
    int cblock = get_global_id(1);
    int n = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(wh, cblock, n);
    int w = wh % shape.w;
    int h = wh / shape.w;
    int c = cblock << 2;
    
    int idx = c * shape.w + w;    // c/4*w
    int idy = n * shape.z + h;    // n*h
    INPUT_TYPE4 in = RI_DATA(input_ptr, SAMPLER, (int2)(idx, idy));

#ifdef USE_IMAGE
    WI_DATA(output_ptr, (int2)(idx, idy), CONVERT_OUTPUT_I4(in));
#else
    #if OUTPUT_FORMAT == MNN_DATA_FORMAT_NCHW
    int output_offset = ((n * shape.y + c) * shape.z + h) * shape.w + w;
    int stride = shape.z * shape.w;
    output_ptr[output_offset] = (OUTPUT_TYPE)in.x;
    output_ptr[output_offset + stride] = (OUTPUT_TYPE)in.y;
    output_ptr[output_offset + stride + stride] = (OUTPUT_TYPE)in.z;
    output_ptr[output_offset + stride + stride + stride] = (OUTPUT_TYPE)in.w;
    #elif OUTPUT_FORMAT == MNN_DATA_FORMAT_NHWC
    int output_offset = ((n * shape.z + h) * shape.w + w) * shape.y + c;
    vstore4(CONVERT_OUTPUT4(in), 0, output_ptr + output_offset);
    #elif OUTPUT_FORMAT == MNN_DATA_FORMAT_NC4HW4
    int output_offset = (((cblock * shape.x + n) * shape.z + h) * shape.w + w) * 4;
    vstore4(CONVERT_OUTPUT4(in), 0, output_ptr + output_offset);
    #endif
#endif
}

__kernel void cl_to_gl(GLOBAL_SIZE_3_DIMS
                                    #ifdef USE_IMAGE
                                    __read_only image2d_t input_ptr,
                                    #else
                                    __global INPUT_TYPE *input_ptr,
                                    #endif
                                    __private const int4 shape, // N C H W
                                    __write_only image2d_t output_ptr
) {

    int wh  = get_global_id(0);
    int cblock = get_global_id(1);
    int n = get_global_id(2);

    DEAL_NON_UNIFORM_DIM3(wh, cblock, n);
    int w = wh % shape.w;
    int h = wh / shape.w;
    int c = cblock << 2;
    
    int idx = c * shape.w + w;    // c/4*w
    int idy = n * shape.z + h;    // n*h
#ifdef USE_IMAGE
    INPUT_TYPE4 in = RI_DATA(input_ptr, SAMPLER, (int2)(idx, idy));
#else
    #if INPUT_FORMAT == MNN_DATA_FORMAT_NCHW
    int input_offset = ((n * shape.y + c) * shape.z + h) * shape.w + w;
    int stride = shape.z * shape.w;
    INPUT_TYPE4 in;
    in.x = input_ptr[input_offset];
    in.y = input_ptr[input_offset + stride];
    in.z = input_ptr[input_offset + stride + stride];
    in.w = input_ptr[input_offset + stride + stride + stride];
    #elif INPUT_FORMAT == MNN_DATA_FORMAT_NHWC
    int input_offset = ((n * shape.z + h) * shape.w + w) * shape.y + c;
    INPUT_TYPE4 in = vload4(0, input_ptr + input_offset);
    #elif INPUT_FORMAT == MNN_DATA_FORMAT_NC4HW4
    int input_offset = (((cblock * shape.x + n) * shape.z + h) * shape.w + w) * 4;
    INPUT_TYPE4 in = vload4(0, input_ptr + input_offset);
    #endif
#endif

    WI_DATA(output_ptr, (int2)(idx, idy), CONVERT_OUTPUT_I4(in));
}
