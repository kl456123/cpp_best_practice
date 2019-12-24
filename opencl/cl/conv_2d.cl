__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void conv2d(__read_only image2d_t image,
                                 __constant float * filter,
                                 __constant float * bias,
                                 __global  float * output,
                                 __private int kernel_size,
                                 __private int pad,
                                 __private int dilation,
                                 __private int stride
                                 ){
// nh, wc/4 * 4
const int out_w_c_idx = get_global_id(0);
const int out_b_h_idx = get_global_id(1);
int out_c_idx = out_c_w_idx%;
int out_w_idx = out_c_w_idx/;
}


__kernel void conv2d_buffer(global const float * input,
                                   global const float * filter,
                                   global const float * bias,
                                   global float * output,
                                   glboal const int num,
                                   global const int kernel_size,
                                   global const int pad,
                                   global const int out_channels,
                                   global const int dilation,
                                   global const int4 inputStride,
                                   global const int4 outputStride;
                                   ){
// filter: C_out*K*K*C_in (c_out, k1, k2, c_in)
// output: N*C_out*H*W (b, c_out, h, w)
// input: N*C_in*H*W (b, c_in, h+k1-K/2, w+k2-K/2)
const int out_ind = get_global_id(0);
; const int batch_block = out_channels * out_height * out_width;
; const int channel_block = out_height * out_width;
int batch_block, channel_block, height_block, width_block = outputStride;
const int out_b_ind = out_ind/batch_block;
out_ind = out_ind%batch_block;
const int out_c_ind = out_ind/channel_block;
out_ind = out_ind%channel_block;
const int out_h_ind = out_ind/height_block;
out_ind = out_ind%height_block;
const int out_w_ind = out_ind/width_block;
int sum = 0;
for(int k1=0;k1<kernel_size;k1++){
        for(int k2=0;k2<kernel_size;k2++){
                for(int c=0;c<in_channels;c++){
                        int filter_index = (((c+out_b_ind*kernel_size)*kernel_size+k2)*in_channels)
                        sum+=filter[filter_index]*input[input_index]+ bias[bias_index];
                    }
                }
            }
            output[out_ind] = sum;
        }
