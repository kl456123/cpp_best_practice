/********
* this file used only to load tensor from cpu to gpu,
* so it cannot be called more than once
*/
// (1, (nhwc)/4, 4)
uniform sampler2D input_image;

// from any to any4
// output shape is equal to input shape
uniform ivec4 output_shape;

#define UP_DIV(x, y) (((x)+(y)-1)/(y))

// (1, nhwc/4, 4)
out vec4 color;

void main(){
    ivec2 pos = ivec2(gl_FragCoord.xy);

    int out_4_dim = UP_DIV(output_shape.w, 4)*4;

    float res[4];
    for(int i=0;i<4;++i){
        int output_4_index = (pos.x+pos.y*MAX_TEXTURE_SIZE)*4+i;
        int index = output_4_index/out_4_dim*output_shape.w+output_4_index%out_4_dim;
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
