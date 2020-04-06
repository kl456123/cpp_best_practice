#include "program.h"
#include <iostream>
#include <fstream>

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
    auto src = std::string(std::istreambuf_iterator<char>(fd),
            (std::istreambuf_iterator<char>()));
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

Status Program::Activate(){
    if(program_id_==0){
        return false;
    }
    glUseProgram(program_id_);
    return status_;
}
