uniform sampler2D input_image;

// some attr params
uniform ivec3 output_shape;
out vec4 color;

#define UP_DIV(x, y) (((x)+(y)-1)/(y))
// color: shape(NH, WC4, 4)
// shape: (height, width, channel)
void main() {
    ivec2 pos = ivec2(gl_FragCoord.xy);
    // decompose output index
    int output_index_y = pos.y%output_shape.x;
    int batch_ind = pos.y/output_shape.x;
    int out_4_ind = pos.x%UP_DIV(output_shape.z, 4);
    int output_index_x = pos.x/UP_DIV(output_shape.z, 4);

    color = vec4(0.0);
}
