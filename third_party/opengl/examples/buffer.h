////////////////////////////////////////////////////
// Buffer is used to intermediate storage to upload
// or download
////////////////////////////////////////////////////

#ifndef BUFFER_H_
#define BUFFER_H_

#include "opengl.h"

class Buffer{
    public:
        Buffer(GLsizeiptr size, GLenum type, GLenum usage);
        ~Buffer();

        // map to copy
        void* Map(GLbitfield bufMask);
        void UnMap();

        // accessor
        GLuint id()const{return id_;}
        GLenum type()const{return type_;}
    private:
        GLuint id_;
        GLsizeiptr size_;
        GLenum type_;
};

class ShaderBuffer: public Buffer{
    public:
        ShaderBuffer(GLsizeiptr size)
            :Buffer(size, GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW){
            }
};


#endif
