precision PRECISION float;
uniform sampler2D input;
uniform sampler2D filter;
uniform sampler2D bias;
uniform int kernel_size;
uniform int stride_size;
uniform int padding;
out float color;
void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    color = 0.0;
    for(int i=0;i<kernel_size;++i){
        for (int j=0;j<kernel_size;++j) {
            float a = texelFetch(A, ivec2(pixel.x+i, pixel.y+j), 0).r;
            float b = texelFetch(B, ivec2(pixel.x+i, pixel.y+j), 0).r;
            color += a * b;
        }
    }

}
