#version 430 core
#define LOCAL_SIZE_X 1
#define LOCAL_SIZE_Y 1
#define FORMAT rgba32f

layout(local_size_x=LOCAL_SIZE_X, local_size_y=LOCAL_SIZE_Y) in;

layout(binding=0) buffer destBuffer{
    float data[];
}output_buffer;

layout(location=1) uniform sampler2D image_input;

layout(location=2) uniform ivec2 image_shape;

void main(){
    ivec2 storePos = ivec2(gl_GlobalInvocationID.xy);

    vec4 color = texelFetch(image_input, storePos, 0);
    output_buffer.data[(storePos.x*image_size.y+storePos.y)*4] = color.r;
    output_buffer.data[(storePos.x*image_size.y+storePos.y)*4+1] = color.g;
    output_buffer.data[(storePos.x*image_size.y+storePos.y)*4+2] = color.b;
    output_buffer.data[(storePos.x*image_size.y+storePos.y)*4+3] = color.a;
}
