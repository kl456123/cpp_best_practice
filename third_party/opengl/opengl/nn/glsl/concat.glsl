// (1, n1, 4)
uniform sampler2D input_image;
// (1, n2, 4)
uniform sampler2D other_image;

uniform int axis;

out vec4 color;
void main(){
    ivec2 pos = ivec2(gl_FragCoord.xy);
    color = texelFetch(input_image, pos, 0);
}
