#version 430 core
layout(local_size_x=1, local_size_y=1) in;
layout(location=0) uniform sampler2D image_input;
layout(binding=1) writeonly buffer destBuffer{
    float data[];
}uOutBuffer;
layout(location=2) uniform ivec2 image_size;

void main(){
    ivec2 storePos = ivec2(gl_GlobalInvocationID.xy);
    /* vec4 color = imageLoad(image_input, storePos); */
    vec4 color = texelFetch(image_input, storePos, 0);
    uOutBuffer.data[(storePos.x*image_size.y+storePos.y)*4] = color.r;
    uOutBuffer.data[(storePos.x*image_size.y+storePos.y)*4+1] = color.g;
    uOutBuffer.data[(storePos.x*image_size.y+storePos.y)*4+2] = color.b;
    uOutBuffer.data[(storePos.x*image_size.y+storePos.y)*4+3] = color.a;
}
