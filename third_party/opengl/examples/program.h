#ifndef PROGRAM_H_
#define PROGRAM_H_
#include <string>
#include "opengl.h"
#include "status.h"


class Program{
    public:
        Program();
        virtual ~Program();
        Program& Attach(const std::string& fname, GLenum type=GL_COMPUTE_SHADER);
        Program& Attach(const char* source, GLenum type=GL_COMPUTE_SHADER);
        const unsigned int program_id(){return program_id_;}
        Status Link();
        Status Activate();
    private:
        Status CreateShader(const char* source, GLenum type, int* out);
        unsigned int program_id_=0;
        Status status_;
};

#endif
