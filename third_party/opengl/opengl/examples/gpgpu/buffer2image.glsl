layout(local_size_x=LOCAL_SIZE_X, local_size_y=LOCAL_SIZE_Y) in;
layout(FORMAT, binding=0) writeonly uniform PRECISION image2D image_output;
layout(binding=1) readonly buffer destBuffer{
    float data[];
}input_buffer;




layout(location=2) uniform ivec2 image_shape;

void main(){
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    int offset = (pos.x*image_shape.y+pos.y)*4;
    vec4 res;
    res.r = input_buffer.data[offset];
    res.g = input_buffer.data[offset+1];
    res.b = input_buffer.data[offset+2];
    res.a = input_buffer.data[offset+3];
    imageStore(image_output, pos, res);
}
