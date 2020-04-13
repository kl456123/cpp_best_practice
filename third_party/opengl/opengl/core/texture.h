#ifndef TEXTURE_H_
#define TEXTURE_H_
#include <vector>
#include "opengl/core/opengl.h"

class Texture{
    public:
        Texture(std::vector<int> dims, GLenum format,
                GLenum target, float* image_data=nullptr);
        GLuint id(){return id_;}
        GLenum target(){return target_;}
        virtual ~Texture();
        std::vector<int>& shape(){return dims_;}
        GLenum format()const{return format_;}
    private:
        GLenum target_;
        GLuint id_;
        GLenum format_;
        std::vector<int> dims_;
};


#endif
