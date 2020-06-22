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

    int out_4_dim = UP_DIV(output_shape.w, 4);
    bool used[4];
    vec4 p[4];
    used[0] = false;
    used[1] = false;
    used[2] = false;
    used[3] = false;
    int output_base = (pos.x+pos.y*MAX_TEXTURE_SIZE);
    int output_index0 = output_base*4;
    int index0 = output_index0%output_shape.w;
    int offset0 = output_index0/output_shape.w*out_4_dim+index0/4;

    float res[4];
    vec4 tmp;
    for(int i=0;i<4;++i){
        int output_index = output_index0+i;
        int index = output_index%output_shape.w;
        int offset = output_index/output_shape.w*out_4_dim+index/4;
        if(!used[offset-offset0]){
            used[offset-offset0] = true;
            p[offset-offset0] = texelFetch(input_image, ivec2(offset%MAX_TEXTURE_SIZE, offset/MAX_TEXTURE_SIZE), 0);
        }
        tmp = p[offset-offset0];
        if(index%4==0){
            res[i] = tmp.x;
        }else if(index%4==1){
            res[i] = tmp.y;
        }else if(index%4==2){
            res[i] = tmp.z;
        }else if(index%4==3){
            res[i] = tmp.w;
        }
    }

    color.x = res[0];
    color.y = res[1];
    color.z = res[2];
    color.w = res[3];
}
