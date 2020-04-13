#ifndef OPENGL_CORE_PROGRAM_H_
#define OPENGL_CORE_PROGRAM_H_
#include <string>
#include "opengl/core/opengl.h"
#include "opengl/utils/status.h"


class Program{
    public:
        Program();
        virtual ~Program();
        Program& AttachFile(const std::string fname, GLenum type=GL_COMPUTE_SHADER);
        Program& AttachSource(const std::string source, GLenum type=GL_COMPUTE_SHADER);
        const unsigned int program_id(){return program_id_;}
        Status Link();
        Status Activate();

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

        void set_buffer(const std::string& name){
        }

        void set_sampler2D(){
        }

        void set_image2D(GLuint id,  int tex_id){
            glActiveTexture(GL_TEXTURE0+tex_id);
            glBindTexture(GL_TEXTURE_2D, id);
        }

    private:
        GLint GetLocation(const std::string& name)const{
            return glGetUniformLocation(program_id_, name.c_str());
        }
        Status CreateShader(const std::string, GLenum type, int* out);
        unsigned int program_id_=0;
        Status status_;
};

#endif
