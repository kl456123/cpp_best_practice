// header definitions
#version 430 core
#define LOCAL_SIZE_X 1
#define LOCAL_SIZE_Y 1
#define FORMAT rgba32f

layout(local_size_x=LOCAL_SIZE_X, local_size_y=LOCAL_SIZE_Y) in;
layout(FORMAT, binding=0) uniform image2D output;
layout(location=1)uniform sampler2D input0;
layout(location=2)uniform sampler2D input1;

// shape parameters
layout(location=3)uniform ivec3 image_shape;

void main(){
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    if(all(lessthan(pos, image_shape))){
        vec4 res = texelFetch(input0, pos, 0) + texelFetch(input1, pos, 1);
        imageStore(output, pos, res);
    }

}

