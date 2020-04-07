#include "texture.h"

Texture::Texture(std::initializer_list<int> dims, GLenum texture_format,
        GLenum target)
:BufferBase(target, texture_format){
    // just 3d texture
    target_ = target;

    glGenTextures(1, &id_);
    glBindTexture(target, id_);
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(target, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    auto dims_start = dims.begin();
    glTexStorage3D(target, 1, type_, dims_start[0], dims_start[1], dims_start[2]/4);
}

Texture::~Texture(){
    glDeleteTextures(1, &id_);
}
