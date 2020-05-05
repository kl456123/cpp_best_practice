uniform sampler2D input0;
uniform sampler2D input1;
uniform ivec2 image_shape;
out vec4 color;
void main() {
    ivec2 pos = ivec2(gl_FragCoord.xy);
    if(all(lessThan(pos, image_shape)))
    {
        color = texelFetch(input0, pos, 0) + texelFetch(input1, pos, 0);
    }
}

