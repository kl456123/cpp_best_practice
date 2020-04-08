#ifndef TEXTURE_H_
#define TEXTURE_H_
#include <vector>
#include "opengl.h"

class Texture{
    public:
        Texture(std::initializer_list<int> dims, GLenum format,
                GLenum target = GL_TEXTURE_3D);
        GLuint id(){return id_;}
        GLenum target(){return target_;}
        virtual ~Texture();
    private:
        GLenum target_;
        GLuint id_;
        GLenum format_;
};


#endif
