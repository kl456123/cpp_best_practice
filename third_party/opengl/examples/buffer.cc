#include "buffer.h"


Buffer::Buffer(GLsizeiptr size, GLenum target, GLenum usage){
    target_ = target;
    size_ = size;

    // create new name of buffer object
    glGenBuffers(1, &id_);
    // assign to context
    glBindBuffer(target, id_);
    // allocate mem in device
    glBufferData(target, size_, NULL, usage);
}

Buffer::~Buffer(){
    glDeleteBuffers(1, &id_);
}


void* Buffer::Map(GLbitfield bufMask){
    glBindBuffer(target_, id_);
    auto ptr = glMapBufferRange(target_, 0, size_, bufMask);
    return ptr;
}


void Buffer::UnMap(){
    glBindBuffer(target_, id_);
    glUnmapBuffer(target_);
}
