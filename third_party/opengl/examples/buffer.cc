#include "buffer.h"


Buffer::Buffer(GLsizeiptr size, GLenum type, GLenum usage){
    type_ = type;
    size_ = size;

    glGenBuffers(1, &id_);
    glBindBuffer(type_, id_);
    // allocate mem in device
    glBufferData(type_, size_, NULL, usage);
}

Buffer::~Buffer(){
    glDeleteBuffers(1, &id_);
}


void* Buffer::Map(GLbitfield bufMask){
    glBindBuffer(type_, id_);
    auto ptr = glMapBufferRange(type_, 0, size_, bufMask);
    return ptr;
}


void Buffer::UnMap(){
    glBindBuffer(type_, id_);
    glUnmapBuffer(type_);
}
