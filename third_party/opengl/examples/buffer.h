////////////////////////////////////////////////////
// Buffer is used to intermediate storage to upload
// or download
////////////////////////////////////////////////////

#ifndef BUFFER_H_
#define BUFFER_H_

#include "buffer_base.h"

class Buffer{
    public:
        Buffer(GLsizeiptr size, GLenum type, GLenum usage);
        ~Buffer();

        // map to copy
        void* Map(GLbitfield bufMask);
        void UnMap();
    private:
        GLsizeiptr size_;

        GLuint id_;
        GLenum target_;


};

class ShaderBuffer: public Buffer{
    public:
        ShaderBuffer(GLsizeiptr size)
            :Buffer(size, GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW){
            }
};


#endif
