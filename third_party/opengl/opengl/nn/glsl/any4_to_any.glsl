// (1, nhwc/4, 4)
uniform sampler2D input_image;

// from any to any4
// output shape is equal to input shape
uniform ivec4 output_shape;

#define UP_DIV(x, y) (((x)+(y)-1)/(y))

// (1, (nhwc)/4, 4)
out vec4 color;

void main(){
    ivec2 pos = ivec2(gl_FragCoord.xy);

    float res[4];
    for(int i=0;i<4;++i){
        int output_index = pos.x*4+i;
        int index = output_index%output_shape.w;
        int offset = output_index/output_shape.w*UP_DIV(output_shape.w, 4)+index/4;
        if(index%4==0){
            res[i] = texelFetch(input_image, ivec2(offset, 0), 0).x;
        }else if(index%4==1){
            res[i] = texelFetch(input_image, ivec2(offset, 0), 0).y;
        }else if(index%4==2){
            res[i] = texelFetch(input_image, ivec2(offset, 0), 0).z;
        }else if(index%4==3){
            res[i] = texelFetch(input_image, ivec2(offset, 0), 0).w;
        }
    }

    color.x = res[0];
    color.y = res[1];
    color.z = res[2];
    color.w = res[3];
}
