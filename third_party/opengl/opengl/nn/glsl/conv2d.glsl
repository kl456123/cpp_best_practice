uniform sampler2D input_image;
uniform sampler2D input_filter;
uniform int kernel_size;
uniform int stride_size;
uniform int padding;
uniform ivec2 image_shape;// height, width
out float color;
void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    color = 0.0;
    for(int i=0;i<kernel_size;++i){
        for (int j=0;j<kernel_size;++j) {
            int input_index_x = pixel.x*stride_size+i-padding;
            int input_index_y = pixel.y*stride_size+j-padding;
            if(input_index_x<0||input_index_x>=image_shape.y){
                continue;
                // when out of boundary
            }
            if(input_index_y<0||input_index_y>=image_shape.x){
                continue;
            }
            vec4 a = texelFetch(input_image, ivec2(input_index_x, input_index_y), 0);
            vec4 b = texelFetch(input_filter, ivec2(i, j), 0);
            color += dot(a , b);
        }
    }

}
