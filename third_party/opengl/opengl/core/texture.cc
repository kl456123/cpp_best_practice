#include "opengl/core/texture.h"

namespace opengl{
    Texture::Texture(std::vector<int> dims, GLenum internal_format,
            GLenum target, float* image_data)
        :dims_(dims){
            // just 3d texture
            target_ = target;

            // automatically activate the texture at the same time
            glGenTextures(1, &id_);
            glBindTexture(target, id_);
            // change internal field for the object
            glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            // STR
            glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(target, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);



            auto dims_start = dims.begin();
            // allocate storage of 3D
            // (TODO which storage to use is better)
            // glTexStorage3D(target, 1, format_, dims_start[0], dims_start[1], dims_start[2]/4);
            //
            // GLenum internal_format = GL_RGBA32F;
            GLenum format = GL_RGBA;

            glTexImage2D(GL_TEXTURE_2D, 0, internal_format, dims[0], dims[1],
                    0, format, GL_FLOAT, image_data);

            // store internal format
            format_ = internal_format;
        }

    Texture::~Texture(){
        glDeleteTextures(1, &id_);
    }
}//namespace opengl
