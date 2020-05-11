uniform sampler2D input_image;
uniform sampler2D input_filter;
// conv2d params
uniform int kernel_size;
uniform int stride_size;
uniform int padding;

// height, width and channel of input and output
uniform ivec3 input_shape;
uniform ivec3 output_shape;
out vec4 color;
#define UP_DIV(x, y) (((x)+(y)-1)/(y))

// filter shape: (h*w, out_4*in_4*in4, out4)
// image shape: (n*h, w*in_4, in4)
// output shape: (n*h, w*out_4, out4)
// where in4=out4 = 4
void main() {
    color = vec4(0.0);
    ivec2 pos = ivec2(gl_FragCoord.xy);
    // decompose pos
    int output_index_y = pos.y%output_shape.x;
    int batch_ind = pos.y/output_shape.x;

    int out_4_ind = pos.y%UP_DIV(output_shape.z, 4);
    int output_index_x = pos.x/output_shape.y;

    for(int i=0;i<kernel_size;++i){
        for (int j=0;j<kernel_size;++j) {
            int input_index_x = output_index_x*stride_size+i-padding;
            int input_index_y = output_index_y*stride_size+j-padding;
            if(input_index_x<0||input_index_x>=input_shape.y){
                continue;
                // when out of boundary
            }
            if(input_index_y<0||input_index_y>=input_shape.x){
                continue;
            }
            // loop in channel dim
            for(int in_4_ind=0;in_4_ind< UP_DIV(input_shape.z, 4);++in_4_ind){
                // get input image
                int input_pos_y = batch_ind*input_shape.x+input_index_y;
                int input_pos_x = input_index_x*UP_DIV(input_shape.z, 4)+in_4_ind;

                // get input filter
                int filter_pos_y = j*kernel_size+i;
                int filter_pos_x = (out_4_ind*UP_DIV(input_shape.z, 4)+in_4_ind)*4;
                vec4 k0 = texelFetch(input_filter, ivec2(filter_pos_x,   filter_pos_y), 0);
                vec4 k1 = texelFetch(input_filter, ivec2(filter_pos_x+1, filter_pos_y), 0);
                vec4 k2 = texelFetch(input_filter, ivec2(filter_pos_x+2, filter_pos_y), 0);
                vec4 k3 = texelFetch(input_filter, ivec2(filter_pos_x+3, filter_pos_y), 0);

                // kernel matrix
                mat4 k = mat4(k0, k1, k2, k3);

                // a 4-elements tuple in output channel dim
                color+=k*texelFetch(input_image, ivec2(input_pos_x, input_pos_y), 0);
            }
        }
    }

}
