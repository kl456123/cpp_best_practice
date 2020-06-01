uniform sampler2D input_image;
uniform vec4 perm;

out vec4 color;
void main(){
    ivec2 pos = ivec2(gl_FragCoord.xy);
    color = texelFetch(input_image, pos, 0);
}
