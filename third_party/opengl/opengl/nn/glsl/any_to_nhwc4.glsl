// (1, (nhwc)/4, 4)
uniform sampler2D input_image;

// from any to any4
// output shape is equal to input shape
uniform ivec4 output_shape;

#define UP_DIV(x, y) (((x)+(y)-1)/(y))

// (nh, wc/4, 4)
out vec4 color;

void main(){
    ivec2 pos = ivec2(gl_FragCoord.xy);

    int out_h_i = pos.y%output_shape.y;
    int out_n_i = pos.y/output_shape.y;
    int out_c_i = pos.x%UP_DIV(output_shape.w, 4);
    int out_w_i = pos.x/UP_DIV(output_shape.w, 4);
    int output_num_elements = output_shape.x * output_shape.y
        * output_shape.z * output_shape.w;

    float res[4];
    for(int i=0;i<4;++i){
        int index = ((out_n_i*output_shape.y+out_h_i)*output_shape.z+out_w_i)*output_shape.w+out_c_i*4+i;
        /* if(index>=output_num_elements){ */
            /* continue; */
        /* } */
        int offset = index/4;
        if(index%4==0){
            res[i] = texelFetch(input_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).x;
        }else if(index%4==1){
            res[i] = texelFetch(input_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).y;
        }else if(index%4==2){
            res[i] = texelFetch(input_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).z;
        }else if(index%4==3){
            res[i] = texelFetch(input_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0).w;
        }
    }

    color.x = res[0];
    color.y = res[1];
    color.z = res[2];
    color.w = res[3];
}
