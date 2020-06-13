// (nh, wc/4, 4)
uniform sampler2D input_image;

// from nhwc4 to any4
// output shape is equal to input shape
uniform ivec4 output_shape;

#define UP_DIV(x, y) (((x)+(y)-1)/(y))

// (1, nhwc/4, 4)
out vec4 color;

void main(){
    ivec2 pos = ivec2(gl_FragCoord.xy);

    int index = pos.x+pos.y*MAX_TEXTURE_SIZE;
    int out_c_i = index % UP_DIV(output_shape.w, 4);
    index = index/UP_DIV(output_shape.w, 4);
    int out_w_i = index%output_shape.z;
    index = index/output_shape.z;
    int out_h_i = index%output_shape.y;
    index = index/output_shape.y;
    int out_n_i = index%output_shape.x;

    // wc/4
    int input_pos_x = out_w_i*UP_DIV(output_shape.w, 4)+out_c_i;
    // nh
    int input_pos_y = out_n_i*output_shape.y+out_h_i;

    color = texelFetch(input_image, ivec2(input_pos_x, input_pos_y), 0);
}
