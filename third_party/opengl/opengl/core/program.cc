#include "program.h"
#include <fstream>
#include <sstream>
#include <glog/logging.h>

Program::~Program(){
    glDeleteProgram(program_id_);
}

Status Program::CreateShader(const std::string source, GLenum type, int* out){
    Status status;
    int shader_id = glCreateShader(type);
    const char* content = source.c_str();
    glShaderSource(shader_id, 1, &content, nullptr);
    glCompileShader(shader_id);
    glGetShaderiv(shader_id, GL_COMPILE_STATUS, &status);
    *out = shader_id;
    return status;
}

Program& Program::AttachSource(const std::string source, GLenum type){
    // Create a Shader Object
    int shader_id;
    Status shader_status = CreateShader(source, type, &shader_id);
    // Display the Build Log on Error
    if (shader_status == false){
        char infoLog[512];
        glGetShaderInfoLog(shader_id, 512, NULL, infoLog);
        LOG(FATAL)<< "ERROR::SHADER::COMPILATION_FAILED\n" << infoLog;
    }
    // Attach the Shader and Free Allocated Memory
    glAttachShader(program_id_, shader_id);
    glDeleteShader(shader_id);
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

Status Program::Link(){
    glLinkProgram(program_id_);
    glGetProgramiv(program_id_, GL_LINK_STATUS, &status_);
    if(status_==false){
        char infoLog[512];
        glGetProgramInfoLog(program_id_, 512, NULL, infoLog);
        LOG(FATAL)<< "ERROR::SHADER::PROGRAM::LINKING_FAILED\n"<<infoLog;
    }
    return status_;
}



Program::Program(){
    program_id_ = glCreateProgram();
}

std::string Program::GetHead(std::string imageFormat) {
    std::ostringstream headOs;
    headOs << "#version 430\n";
    headOs << "#define FORMAT " << imageFormat << "\n";
    headOs << "#define PRECISION mediump\n";
    headOs << "precision PRECISION float;\n";
    headOs<<"#define LOCAL_SIZE_X 1\n";
    headOs<<"#define LOCAL_SIZE_Y 1\n";
    return headOs.str();
}

Status Program::Activate(){
    if(program_id_==0){
        LOG(ERROR)<<"It cannot be activated when program id is zero";
        return false;
    }
    glUseProgram(program_id_);
    return status_;
}
