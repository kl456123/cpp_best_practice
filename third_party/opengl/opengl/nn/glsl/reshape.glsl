uniform sampler2D input_image;


/* uniform ivec4 input_stride; */
/* uniform ive4 output_stride; */
uniform ivec4 input_shape;
uniform ivec4 output_shape;
#define UP_DIV(x, y) (((x)+(y)-1)/(y))
#define UP_ROUND(x, y) (((x) + (y) - (1)) / (y) * (y))

out vec4 color;
void main(){
    ivec2 pos = ivec2(gl_FragCoord.xy);
    int out_dim4 = UP_ROUND(output_shape.w, 4);
    int in_dim4 = UP_ROUND(input_shape.w, 4);

    {
        int output_index = (pos.x*4+0)/out_dim4*output_shape.w+ (pos.x*4+0)%out_dim4;
        int tmp = output_index%input_shape.w;
        int offset = output_index/input_shape.w*UP_DIV(input_shape.w, 4)+tmp/4;
        float value;
        if(tmp%4==0){
            value = texelFetch(input_image, ivec2(offset, 0), 0).x;
        }else if(tmp%4==1){
            value = texelFetch(input_image, ivec2(offset, 0), 0).y;
        }else if(tmp%4==2){
            value = texelFetch(input_image, ivec2(offset, 0), 0).z;
        }else if(tmp%4==3){
            value = texelFetch(input_image, ivec2(offset, 0), 0).w;
        }
        color.x = value;
    }


    {
        int output_index = (pos.x*4+1)/out_dim4*output_shape.w+ (pos.x*4+1)%out_dim4;
        int tmp = output_index%input_shape.w;
        int offset = output_index/input_shape.w*UP_DIV(input_shape.w, 4)+tmp/4;
        float value;
        if(tmp%4==0){
            value = texelFetch(input_image, ivec2(offset, 0), 0).x;
        }else if(tmp%4==1){
            value = texelFetch(input_image, ivec2(offset, 0), 0).y;
        }else if(tmp%4==2){
            value = texelFetch(input_image, ivec2(offset, 0), 0).z;
        }else if(tmp%4==3){
            value = texelFetch(input_image, ivec2(offset, 0), 0).w;
        }
        color.y = value;
    }

    {
        int output_index = (pos.x*4+2)/out_dim4*output_shape.w+ (pos.x*4+2)%out_dim4;
        int tmp = output_index%input_shape.w;
        int offset = output_index/input_shape.w*UP_DIV(input_shape.w, 4)+tmp/4;
        float value;
        if(tmp%4==0){
            value = texelFetch(input_image, ivec2(offset, 0), 0).x;
        }else if(tmp%4==1){
            value = texelFetch(input_image, ivec2(offset, 0), 0).y;
        }else if(tmp%4==2){
            value = texelFetch(input_image, ivec2(offset, 0), 0).z;
        }else if(tmp%4==3){
            value = texelFetch(input_image, ivec2(offset, 0), 0).w;
        }
        color.z = value;
    }

    {
        int output_index = (pos.x*4+3)/out_dim4*output_shape.w+ (pos.x*4+3)%out_dim4;
        int tmp = output_index%input_shape.w;
        int offset = output_index/input_shape.w*UP_DIV(input_shape.w, 4)+tmp/4;
        float value;
        if(tmp%4==0){
            value = texelFetch(input_image, ivec2(offset, 0), 0).x;
        }else if(tmp%4==1){
            value = texelFetch(input_image, ivec2(offset, 0), 0).y;
        }else if(tmp%4==2){
            value = texelFetch(input_image, ivec2(offset, 0), 0).z;
        }else if(tmp%4==3){
            value = texelFetch(input_image, ivec2(offset, 0), 0).w;
        }
        color.w = value;
    }
}
