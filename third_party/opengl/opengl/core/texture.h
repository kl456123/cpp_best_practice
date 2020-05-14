#ifndef TEXTURE_H_
#define TEXTURE_H_
#include <vector>
#include <glog/logging.h>
#include "opengl/core/opengl.h"

namespace opengl{
    class Texture{
        public:
            Texture(std::vector<int> dims, GLenum format,
                    GLenum target, float* image_data=nullptr);
            GLuint id(){return id_;}
            GLenum target(){return target_;}
            virtual ~Texture();
            std::vector<int>& shape(){return dims_;}
            const int height()const{
                CHECK_EQ(dims_.size(), 2);
                return dims_[1];
            }
            const int width()const{
                CHECK_EQ(dims_.size(), 2);
                return dims_[0];
            }
            GLenum format()const{return format_;}
        private:
            GLenum target_;
            GLuint id_;
            GLenum format_;
            // width, height
            std::vector<int> dims_;
    };
}//namespace opengl

#endif
