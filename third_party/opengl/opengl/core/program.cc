#include "opengl/core/program.h"
#include "opengl/utils/macros.h"
#include "opengl/core/tensor.h"
#include "opengl/core/driver.h"
#include <fstream>
#include <sstream>
#include <glog/logging.h>

namespace opengl{
    namespace{
        struct Vertex {
            float x, y;
        };
    }
    Program::~Program(){
        OPENGL_CALL(glDeleteProgram(program_id_));
    }

    OGLStatus Program::CreateShader(const std::string source, GLenum type, int* out){
        OGLStatus status;
        int shader_id = glCreateShader(type);
        const char* content = source.c_str();
        OPENGL_CALL(glShaderSource(shader_id, 1, &content, nullptr));
        OPENGL_CALL(glCompileShader(shader_id));
        OPENGL_CALL(glGetShaderiv(shader_id, GL_COMPILE_STATUS, &status));
        *out = shader_id;
        return status;
    }

    Program& Program::AttachShader(const GLuint shader_id){
        OPENGL_CALL(glAttachShader(program_id_, shader_id));
        return *this;
    }

    Program& Program::AttachSource(const std::string source, GLenum type){
        // Create a Shader Object
        int shader_id;
        OGLStatus shader_status = CreateShader(source, type, &shader_id);
        // Display the Build Log on Error
        if (!shader_status){
            char infoLog[512];
            glGetShaderInfoLog(shader_id, 512, NULL, infoLog);
            LOG(FATAL)<< "ERROR::SHADER::COMPILATION_FAILED\n" << infoLog;
        }
        // Attach the Shader and Free Allocated Memory
        OPENGL_CALL(glAttachShader(program_id_, shader_id));
        OPENGL_CALL(glDeleteShader(shader_id));
        return *this;
    }

    Program& Program::AttachFile(const std::string fname, GLenum type){
        // Load GLSL Shader Source from File
        std::ifstream fd(fname);
        std::string src = std::string(std::istreambuf_iterator<char>(fd),
                (std::istreambuf_iterator<char>()));
        if(src.empty()){
            LOG(FATAL)<<"Read File ERROR from "<<fname;
        }
        // add head
        std::ostringstream tc;
        tc << GetHead("rgba32f");
        tc<<src;

        return AttachSource(tc.str().c_str(), type);
    }

    OGLStatus Program::Link(){
        OPENGL_CALL(glLinkProgram(program_id_));
        OPENGL_CALL(glGetProgramiv(program_id_, GL_LINK_STATUS, &status_));
        if(status_==false){
            char infoLog[512];
            OPENGL_CALL(glGetProgramInfoLog(program_id_, 512, NULL, infoLog));
            LOG(FATAL)<< "ERROR::SHADER::PROGRAM::LINKING_FAILED\n"<<infoLog;
        }
        return status_;
    }



    Program::Program(){
        program_id_ = glCreateProgram();
    }

    std::string Program::GetHead(std::string imageFormat) {
        std::ostringstream headOs;
        headOs << "#version "<<GLSL_VERSION<<"\n";
        headOs << "#define FORMAT " << imageFormat << "\n";
        headOs << "#define PRECISION highp\n";
        headOs << "precision PRECISION float;\n";
        headOs<<"#define LOCAL_SIZE_X 1\n";
        headOs<<"#define LOCAL_SIZE_Y 1\n";
        return headOs.str();
    }

    OGLStatus Program::Activate(){
        if(program_id_==0){
            LOG(FATAL)<<"It cannot be activated when program id is zero";
            return false;
        }
        OPENGL_CALL(glUseProgram(program_id_));
        return status_;
    }

    void Program::SetRetVal(const TensorList& outputs){
        CHECK_EQ(outputs.size(), 1);
        CHECK_EQ(outputs[0]->mem_type(), Tensor::DEVICE_TEXTURE);

        const int width = outputs[0]->device<Texture>()->shape()[0];
        const int height = outputs[0]->device<Texture>()->shape()[1];
        auto output_texture = outputs[0]->device<Texture>()->id();

        AttachTextureToFrameBuffer(output_texture, width, height);
    }

    void Program::Run(bool sync){
        OPENGL_CALL(glClear(GL_COLOR_BUFFER_BIT));
        OPENGL_CALL(glDrawArrays(GL_TRIANGLES, 0, 6));
        if(sync){
            OPENGL_CALL(glFinish());
        }
    }

    void Program::SetVertexShader(){
        // set input arguments for vertex shader
        auto point_attrib = GLuint(glGetAttribLocation(program_id(), "point"));
        OPENGL_CHECK_ERROR;
        OPENGL_CALL(glEnableVertexAttribArray(point_attrib));
        OPENGL_CALL(glVertexAttribPointer(point_attrib, 2, GL_FLOAT, GL_FALSE,
                    sizeof(Vertex), nullptr));
    }
}//namespace opengl
