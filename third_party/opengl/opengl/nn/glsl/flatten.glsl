uniform sampler2D input_image;

// shape: (height, width, channel)
uniform ivec3 input_shape;
uniform ivec3 output_shape;
out vec4 color;

#define UP_DIV(x, y) (((x)+(y)-1)/(y))
// input internal shape: (n*h, w*in_4, in4)
// output internal shape: (n*h, w*out_4, out4)
void main() {
    ivec2 pos = ivec2(gl_FragCoord.xy);
    // decompose output index
    int output_index_y = pos.y%output_shape.x;
    int batch_ind = pos.y/output_shape.x;
    int out_4_ind = pos.x%UP_DIV(output_shape.z, 4);
    int output_index_x = pos.x/UP_DIV(output_shape.z, 4);

    int total_size = input_shape.x*input_shape.y*input_shape.z;
    int output_channel_index = out_4_ind*4;

    // compose output index use output shape
    int offset_base = ((batch_ind*output_shape.x+output_index_y)*output_shape.y+output_index_x)*
        output_shape.z+output_channel_index;
    color = vec4(0.0);
    /* color = texelFetch(input_image, pos, 0); */
    // input index equals to output index
    for(int i=0;i<4;++i){
        int offset0 = offset_base+i;
        if(offset0>=total_size){
            continue;
        }

        // decompose input offset use input shape
        int input_channel_index = offset0%input_shape.z;
        offset0 = offset0/input_shape.z;
        int input_index_x = offset0%input_shape.y;
        offset0 = offset0/input_shape.y;
        int input_index_y = offset0%input_shape.x;
        offset0 = offset0/input_shape.x;
        int input_batch_ind = offset0;

        // compose it to x, y
        // input internal shape: (n*h, w*in_4, in4)
        int input_pos_x = input_index_x*UP_DIV(input_shape.z, 4)+input_channel_index/4;
        int input_pos_y = input_batch_ind*input_shape.x+input_index_y;
        int input_pos_z = input_channel_index%4;
        if(input_pos_z==0){
            color.x = texelFetch(input_image, ivec2(input_pos_x, input_pos_y), 0).r;
        }else if(input_pos_z==1){
            color.y = texelFetch(input_image, ivec2(input_pos_x, input_pos_y), 0).g;
        }else if(input_pos_z==2){
            color.z = texelFetch(input_image, ivec2(input_pos_x, input_pos_y), 0).b;
        }else{
            color.w = texelFetch(input_image, ivec2(input_pos_x, input_pos_y), 0).a;
        }

    }
}
