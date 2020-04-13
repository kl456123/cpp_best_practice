layout(local_size_x=LOCAL_SIZE_X, local_size_y=LOCAL_SIZE_Y) in;

layout(FORMAT, binding=0) writeonly uniform PRECISION image2D image_output;
layout(location=1) uniform mediump sampler2D input0;
layout(location=2) uniform mediump sampler2D input1;
layout(location=3) uniform ivec2 image_shape;

void main()
{
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    if(all(lessThan(pos, image_shape)))
    {
        vec4 sum = texelFetch(input0, pos, 0) + texelFetch(input1, pos, 0);
        imageStore(image_output, pos, sum);
    }
}
