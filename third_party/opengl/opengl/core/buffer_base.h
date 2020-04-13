#ifndef BUFFER_BASE_H_
#define BUFFER_BASE_H_
#include "opengl.h"

class BufferBase{
    public:
        BufferBase(GLuint id, GLenum target);
        virtual ~BufferBase(){};
        // accessor
        GLuint id()const{return id_;}
        GLenum target()const{return target_;}
    protected:
        GLuint id_;
        GLenum target_;
};


#endif
