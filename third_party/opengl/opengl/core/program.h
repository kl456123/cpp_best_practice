#ifndef OPENGL_CORE_PROGRAM_H_
#define OPENGL_CORE_PROGRAM_H_
#include <string>
#include "opengl/core/opengl.h"
#include "opengl/utils/status.h"

namespace opengl{

    class Program{
        public:
            Program();
            virtual ~Program();
            Program& AttachFile(const std::string fname, GLenum type=GL_COMPUTE_SHADER);
            Program& AttachSource(const std::string source, GLenum type=GL_COMPUTE_SHADER);
            const unsigned int program_id(){return program_id_;}
            OGLStatus Link();
            OGLStatus Activate();

            std::string GetHead(std::string imageFormat="rgba32f");
            GLuint program_id()const{return program_id_;}

            // utils for set params
            void set_bool(const std::string& name, bool value)const{
                glUniform1i(GetLocation(name), (int)value);
            }

            void set_int(const std::string& name, int value)const{
                glUniform1i(GetLocation(name), value);
            }

            void set_float(const std::string& name, float value)const{
                glUniform1f(GetLocation(name), value);
            }

            void set_vec2(const std::string& name, float x, float y)const{
                glUniform2f(GetLocation(name), x, y);
            }

            void set_vec2i(const std::string& name, int x, int y)const{
                glUniform2i(GetLocation(name), x, y);
            }

            void set_buffer(GLuint SSBO, GLenum target){
                glBindBufferBase(target, 1, SSBO);
            }

            void set_input_sampler2D(GLuint tex_output, GLenum internal_format){
                glBindImageTexture(0, tex_output, 0, GL_TRUE, 0, GL_READ_ONLY, internal_format);
            }

            void set_output_sampler2D(GLuint tex_output, GLenum internal_format){
                glBindImageTexture(0, tex_output, 0, GL_TRUE, 0, GL_WRITE_ONLY, internal_format);
            }

            void set_image2D(const std::string& name, GLuint id,  int tex_id){
                set_int(name, tex_id);
                glActiveTexture(GL_TEXTURE0+tex_id);
                glBindTexture(GL_TEXTURE_2D, id);
            }

        private:
            GLint GetLocation(const std::string& name)const{
                return glGetUniformLocation(program_id_, name.c_str());
            }
            OGLStatus CreateShader(const std::string, GLenum type, int* out);
            unsigned int program_id_=0;
            OGLStatus status_;
    };

}//namespace opengl

#endif
