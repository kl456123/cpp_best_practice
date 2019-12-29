__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void conv2d_buffer(global const float *input,
                            global const float *filter,
                            global const float *bias,
                            global float *output,
                            int kernel_size,
                            int dilation,
                            int stride,
                            int4 inputStride,
                            int4 outputStride,
                            int2 inputShape 
                            )
{
    // filter: C_out*K*K*C_in (c_out, k1, k2, c_in)
    // output: N*C_out*H*W (b, c_out, h, w)
    // input: N*C_in*H*W (b, c_in, h+k1-K/2, w+k2-K/2)
    // bias: N*C_out

//     kernel_size = kernel_size * dilation - 1;
//     int out_channels = outputStride.x/outputStride.y;
//     int in_channels = inputStride.x/inputStride.y;

//     int out_ind = get_global_id(0);
//     int batch_block = outputStride.x;
//     int channel_block = outputStride.y;
//     int height_block = outputStride.z;
//     int width_block = outputStride.w;
//     int out_b_ind = out_ind / batch_block;
//     out_ind = out_ind % batch_block;
//     int out_c_ind = out_ind / channel_block;
//     out_ind = out_ind % channel_block;
//     int out_h_ind = out_ind / height_block;
//     out_ind = out_ind % height_block;
//     int out_w_ind = out_ind / width_block;
//     float sum = 0;
//     for (int k1 = 0; k1 < kernel_size; k1++)
//     {
//         for (int k2 = 0; k2 < kernel_size; k2++)
//         {
//             for (int in_c_ind = 0; in_c_ind < in_channels; in_c_ind++)
//             {
//                 int filter_index = ((out_c_ind * kernel_size + k1) * kernel_size + k2) * in_channels + in_c_ind;
//                 int in_h_ind = out_h_ind * stride;
//                 int in_w_ind = out_w_ind * stride;
//                 // check insane
//                 if (in_h_ind < 0 || in_h_ind >= inputShape.x || in_w_ind < 0 || in_w_ind >= inputShape.y)
//                 {
//                     continue;
//                 }
//                 int input_index = out_b_ind * inputStride.x + in_c_ind * inputStride.y +
//                                   in_h_ind * inputStride.z + in_w_ind * inputStride.w;
//                 int bias_index = out_c_ind+ out_b_ind*out_channels;
//                 sum += filter[filter_index] * input[input_index] + bias[bias_index];
//             }
//         }
//     }
//     output[out_ind] = sum;
int out_ind = get_global_id(0);
if(out_ind<100){
output[out_ind]= input[out_ind];
}
}
