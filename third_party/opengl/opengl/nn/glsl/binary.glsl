#version 430
layout(rgba32f, binding=0) uniform image3D uOutput;
layout(location=1) uniform sampler3D uInput0;
layout(location=2) uniform sampler3D uInput1;
layout(location=3) uniform ivec4 imgSize;
layout(local_size_x = 8, local_size_y = 8, local_size_z=1) in;
void main() {
    ivec3 pos = ivec3(gl_GlobalInvocationID);
    ivec3 inSize = imgSize.xyz;
    vec4 sum = texelFetch(uInput0, pos, 0) + texelFetch(uInput1, pos, 0);
    imageStore(uOutput, pos, sum);
}