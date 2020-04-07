#ifndef BUFFER_BASE_H_
#define BUFFER_BASE_H_
#include "opengl.h"

class BufferBase{
    BufferBase();
// accessor
        GLuint id()const{return id_;}
        GLenum type()const{return type_;}
    protected:
        GLuint id_;
        GLenum type_;
};


#endif
