layout(local_size_x=LOCAL_SIZE_X, local_size_y=LOCAL_SIZE_Y) in;

layout(FORMAT, binding=0) readonly uniform PRECISION image2D image_input;
layout(binding=1) writeonly buffer destBuffer{
    float data[];
}output_buffer;


layout(location=2) uniform ivec2 image_shape;

void main(){
    ivec2 storePos = ivec2(gl_GlobalInvocationID.xy);

    vec4 color = imageLoad(image_input, storePos);
    output_buffer.data[(storePos.x*image_shape.y+storePos.y)*4] = color.r;
    output_buffer.data[(storePos.x*image_shape.y+storePos.y)*4+1] = color.g;
    output_buffer.data[(storePos.x*image_shape.y+storePos.y)*4+2] = color.b;
    output_buffer.data[(storePos.x*image_shape.y+storePos.y)*4+3] = color.a;
}
