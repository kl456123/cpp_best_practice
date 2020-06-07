uniform sampler2D input_image;
uniform sampler2D input_filter;
uniform sampler2D input_bias;
// conv2d params
uniform int stride_size;
uniform int kernel_size;
uniform int group;
uniform int dilation;
uniform int padding;
// use int type to represent bool type
uniform int use_bias;

// height, width and channel of input and output
uniform ivec3 input_shape;
uniform ivec3 output_shape;
out vec4 color;
#define UP_DIV(x, y) (((x)+(y)-1)/(y))

// filter shape: (h*w* out_4, in_4*in4, out4)
// image shape: (n*h, w*in_4, in4)
// output shape: (n*h, w*out_4, out4)
// bias shape: (n*1, 1*out_4, out4)
// where in4=out4 = 4
void main() {
    ivec2 pos = ivec2(gl_FragCoord.xy);
    // decompose pos
    // pos = (w*out_4_i, nh_i)
    // output_shape=(h,w,c)
    int output_index_y = pos.y%output_shape.x;
    int batch_ind = pos.y/output_shape.x;

    int out_4_ind = pos.x%UP_DIV(output_shape.z, 4);
    int output_index_x = pos.x/UP_DIV(output_shape.z, 4);




    int bias_pos_x = out_4_ind;
    int bias_pos_y = batch_ind;

    int out_group_size = output_shape.z/group;

    int in_group_size = input_shape.z/group;

    if(use_bias==1){
        color = texelFetch(input_bias, ivec2(bias_pos_x,   bias_pos_y), 0);
    }else{
        color = vec4(0.0);
    }

    {
        int out_c_ind = out_4_ind*4+0;
        int grp_ind = out_c_ind/group;
        float value = 0.0;

        for(int i=0;i<kernel_size;++i){
            for (int j=0;j<kernel_size;++j) {
                int input_index_x = output_index_x*stride_size+i*dilation-padding;
                int input_index_y = output_index_y*stride_size+j*dilation-padding;
                if(input_index_x<0||input_index_x>=input_shape.y){
                    continue;
                    // when out of boundary
                }
                if(input_index_y<0||input_index_y>=input_shape.x){
                    continue;
                }
                int input_spatial_index = input_index_x + input_index_y*input_shape.y;
                int filter_spatial_index = i+j*kernel_size;
                for(int k=0;k<in_group_size;++k){
                    int in_c_ind = k+grp_ind*in_group_size;
                    int input_index = input_spatial_index*input_shape.z+in_c_ind;
                    int filter_in_c_ind = in_c_ind;
                    int filter_out_c_ind = out_c_ind;
                    // load filter and input image according to the index
                    // w*in_4
                    int input_pos_x = input_index_x * UP_DIV(input_shape.z, 4)+ in_c_ind/4;
                    // n*h
                    int input_pos_y = batch_ind*input_shape.x+input_index_y;
                    vec4 input_vec = texelFetch(input_image, ivec2(input_pos_x, input_pos_y), 0);
                    float input_value;
                    if(in_c_ind%4==0){
                        input_value = input_vec.x;
                    }else if(in_c_ind%4==1){
                        input_value = input_vec.y;
                    }else if(in_c_ind%4==2){
                        input_value = input_vec.z;
                    }else if(in_c_ind%4==3){
                        input_value = input_vec.w;
                    }
                    // in_4*in4
                    int filter_pos_x = k;
                    // h * w * out_4
                    int filter_pos_y = filter_spatial_index*UP_DIV(output_shape.z, 4)+out_c_ind/4;
                    vec4 filter_vec = texelFetch(input_filter, ivec2(filter_pos_x, filter_pos_y), 0);
                    float filter_value;
                    if(out_c_ind%4==0){
                        filter_value = filter_vec.x;
                    }else if(out_c_ind%4==1){
                        filter_value = filter_vec.y;
                    }else if(out_c_ind%4==2){
                        filter_value = filter_vec.z;
                    }else if(out_c_ind%4==3){
                        filter_value = filter_vec.w;
                    }
                    value+=input_value*filter_value;
                }
            }
        }
        color.x += value;
    }

    {
        int out_c_ind = out_4_ind*4+1;
        int grp_ind = out_c_ind/group;
        float value = 0.0;

        for(int i=0;i<kernel_size;++i){
            for (int j=0;j<kernel_size;++j) {
                int input_index_x = output_index_x*stride_size+i*dilation-padding;
                int input_index_y = output_index_y*stride_size+j*dilation-padding;
                if(input_index_x<0||input_index_x>=input_shape.y){
                    continue;
                    // when out of boundary
                }
                if(input_index_y<0||input_index_y>=input_shape.x){
                    continue;
                }
                int input_spatial_index = input_index_x + input_index_y*input_shape.y;
                int filter_spatial_index = i+j*kernel_size;
                for(int k=0;k<in_group_size;++k){
                    int in_c_ind = k+grp_ind*in_group_size;
                    int input_index = input_spatial_index*input_shape.z+in_c_ind;
                    int filter_in_c_ind = in_c_ind;
                    int filter_out_c_ind = out_c_ind;
                    // load filter and input image according to the index
                    // w*in_4
                    int input_pos_x = input_index_x * UP_DIV(input_shape.z, 4)+ in_c_ind/4;
                    // n*h
                    int input_pos_y = batch_ind*input_shape.x+input_index_y;
                    vec4 input_vec = texelFetch(input_image, ivec2(input_pos_x, input_pos_y), 0);
                    float input_value;
                    if(in_c_ind%4==0){
                        input_value = input_vec.x;
                    }else if(in_c_ind%4==1){
                        input_value = input_vec.y;
                    }else if(in_c_ind%4==2){
                        input_value = input_vec.z;
                    }else if(in_c_ind%4==3){
                        input_value = input_vec.w;
                    }
                    // in_4*in4
                    int filter_pos_x = k;
                    // h * w * out_4
                    int filter_pos_y = filter_spatial_index*UP_DIV(output_shape.z, 4)+out_c_ind/4;
                    vec4 filter_vec = texelFetch(input_filter, ivec2(filter_pos_x, filter_pos_y), 0);
                    float filter_value;
                    if(out_c_ind%4==0){
                        filter_value = filter_vec.x;
                    }else if(out_c_ind%4==1){
                        filter_value = filter_vec.y;
                    }else if(out_c_ind%4==2){
                        filter_value = filter_vec.z;
                    }else if(out_c_ind%4==3){
                        filter_value = filter_vec.w;
                    }
                    value+=input_value*filter_value;
                }
            }
        }
        color.y += value;
    }

    {
        int out_c_ind = out_4_ind*4+2;
        int grp_ind = out_c_ind/group;
        float value = 0.0;

        for(int i=0;i<kernel_size;++i){
            for (int j=0;j<kernel_size;++j) {
                int input_index_x = output_index_x*stride_size+i*dilation-padding;
                int input_index_y = output_index_y*stride_size+j*dilation-padding;
                if(input_index_x<0||input_index_x>=input_shape.y){
                    continue;
                    // when out of boundary
                }
                if(input_index_y<0||input_index_y>=input_shape.x){
                    continue;
                }
                int input_spatial_index = input_index_x + input_index_y*input_shape.y;
                int filter_spatial_index = i+j*kernel_size;
                for(int k=0;k<in_group_size;++k){
                    int in_c_ind = k+grp_ind*in_group_size;
                    int input_index = input_spatial_index*input_shape.z+in_c_ind;
                    int filter_in_c_ind = in_c_ind;
                    int filter_out_c_ind = out_c_ind;
                    // load filter and input image according to the index
                    // w*in_4
                    int input_pos_x = input_index_x * UP_DIV(input_shape.z, 4)+ in_c_ind/4;
                    // n*h
                    int input_pos_y = batch_ind*input_shape.x+input_index_y;
                    vec4 input_vec = texelFetch(input_image, ivec2(input_pos_x, input_pos_y), 0);
                    float input_value;
                    if(in_c_ind%4==0){
                        input_value = input_vec.x;
                    }else if(in_c_ind%4==1){
                        input_value = input_vec.y;
                    }else if(in_c_ind%4==2){
                        input_value = input_vec.z;
                    }else if(in_c_ind%4==3){
                        input_value = input_vec.w;
                    }
                    // in_4*in4
                    int filter_pos_x = k;
                    // h * w * out_4
                    int filter_pos_y = filter_spatial_index*UP_DIV(output_shape.z, 4)+out_c_ind/4;
                    vec4 filter_vec = texelFetch(input_filter, ivec2(filter_pos_x, filter_pos_y), 0);
                    float filter_value;
                    if(out_c_ind%4==0){
                        filter_value = filter_vec.x;
                    }else if(out_c_ind%4==1){
                        filter_value = filter_vec.y;
                    }else if(out_c_ind%4==2){
                        filter_value = filter_vec.z;
                    }else if(out_c_ind%4==3){
                        filter_value = filter_vec.w;
                    }
                    value+=input_value*filter_value;
                }
            }
        }
        color.z += value;
        /* color.z = float(out_c_ind); */
    }

    {
        int out_c_ind = out_4_ind*4+3;
        int grp_ind = out_c_ind/group;
        float value = 0.0;

        for(int i=0;i<kernel_size;++i){
            for (int j=0;j<kernel_size;++j) {
                int input_index_x = output_index_x*stride_size+i*dilation-padding;
                int input_index_y = output_index_y*stride_size+j*dilation-padding;
                if(input_index_x<0||input_index_x>=input_shape.y){
                    continue;
                    // when out of boundary
                }
                if(input_index_y<0||input_index_y>=input_shape.x){
                    continue;
                }
                int input_spatial_index = input_index_x + input_index_y*input_shape.y;
                int filter_spatial_index = i+j*kernel_size;
                for(int k=0;k<in_group_size;++k){
                    int in_c_ind = k+grp_ind*in_group_size;
                    int input_index = input_spatial_index*input_shape.z+in_c_ind;
                    int filter_in_c_ind = in_c_ind;
                    int filter_out_c_ind = out_c_ind;
                    // load filter and input image according to the index
                    // w*in_4
                    int input_pos_x = input_index_x * UP_DIV(input_shape.z, 4)+ in_c_ind/4;
                    // n*h
                    int input_pos_y = batch_ind*input_shape.x+input_index_y;
                    vec4 input_vec = texelFetch(input_image, ivec2(input_pos_x, input_pos_y), 0);
                    float input_value;
                    if(in_c_ind%4==0){
                        input_value = input_vec.x;
                    }else if(in_c_ind%4==1){
                        input_value = input_vec.y;
                    }else if(in_c_ind%4==2){
                        input_value = input_vec.z;
                    }else if(in_c_ind%4==3){
                        input_value = input_vec.w;
                    }
                    // in_4*in4
                    int filter_pos_x = k;
                    // h * w * out_4
                    int filter_pos_y = filter_spatial_index*UP_DIV(output_shape.z, 4)+out_c_ind/4;
                    vec4 filter_vec = texelFetch(input_filter, ivec2(filter_pos_x, filter_pos_y), 0);
                    float filter_value;
                    if(out_c_ind%4==0){
                        filter_value = filter_vec.x;
                    }else if(out_c_ind%4==1){
                        filter_value = filter_vec.y;
                    }else if(out_c_ind%4==2){
                        filter_value = filter_vec.z;
                    }else if(out_c_ind%4==3){
                        filter_value = filter_vec.w;
                    }
                    value+=input_value*filter_value;
                }
            }
        }
        color.w += value;
    }
}
