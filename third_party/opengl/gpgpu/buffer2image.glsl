#version 430 core
#define LOCAL_SIZE_X 1
#define LOCAL_SIZE_Y 1
#define FORMAT rgba32f

layout(local_size_x=LOCAL_SIZE_X, local_size_y=LOCAL_SIZE_Y) in;

layout(binding=0) uniform sampler2D image_output;

layout(location=1) buffer destBuffer{
    float data[];
}input_buffer;

layout(location=2) uniform ivec2 image_shape;

void main(){
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    int offset = (pos.x*image_shape.y+pos.y)*4;
    vec4 res;
    res.r = destBuffer.data[offset];
    res.g = destBuffer.data[offset+1];
    res.b = destBuffer.data[offset+2];
    res.a = destBuffer.data[offset+3];
    imageStore(image_output, pos, res);
}
