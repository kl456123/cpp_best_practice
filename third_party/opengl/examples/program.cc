#include "program.h"
#include <iostream>
#include <fstream>
#include <sstream>

Program::~Program(){
    glDeleteProgram(program_id_);
}

Status Program::CreateShader(const char* source, GLenum type, int* out){
    Status status;
    int shader_id = glCreateShader(type);
    glShaderSource(shader_id, 1, &source, nullptr);
    glCompileShader(shader_id);
    glGetShaderiv(shader_id, GL_COMPILE_STATUS, &status);
    *out = shader_id;
    return status;
}

Program& Program::Attach(const char* source, GLenum type){
    // Create a Shader Object
    int shader_id;
    Status shader_status = CreateShader(source, type, &shader_id);
    // Display the Build Log on Error
    if (shader_status == false){
        char infoLog[512];
        glGetShaderInfoLog(shader_id, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    // Attach the Shader and Free Allocated Memory
    glAttachShader(program_id_, shader_id);
    glDeleteShader(shader_id);
    return *this;
}

Program& Program::Attach(const std::string& fname, GLenum type){
    // Load GLSL Shader Source from File
    std::ifstream fd(fname);
    std::string src = std::string(std::istreambuf_iterator<char>(fd),
            (std::istreambuf_iterator<char>()));
    // add head
    // std::ostringstream tc;
    // tc << GetHead();
    // tc<<src;
    const char * source = src.c_str();
    return Attach(source, type);
}

Status Program::Link(){
    glLinkProgram(program_id_);
    glGetProgramiv(program_id_, GL_LINK_STATUS, &status_);
    if(status_==false){
        char infoLog[512];
        glGetProgramInfoLog(program_id_, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    return status_;
}



Program::Program(){
    program_id_ = glCreateProgram();
}

std::string Program::GetHead(std::string imageFormat) {
    std::ostringstream headOs;
    headOs << "#version 310 es\n";
    headOs << "#define PRECISION mediump\n";
    headOs << "precision PRECISION float;\n";
    headOs << "#define FORMAT " << imageFormat << "\n";
    return headOs.str();
}

Status Program::Activate(){
    if(program_id_==0){
        return false;
    }
    glUseProgram(program_id_);
    return status_;
}
