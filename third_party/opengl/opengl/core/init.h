#ifndef OPENGL_CORE_INIT_H_
#define OPENGL_CORE_INIT_H_
#include "opengl.h"

namespace opengl{
    // init some context ,windows and loaded function
    int glut_init(int argc, char* argv[]);
    int glew_init();

#ifdef ARM_PLATFORM
#include <EGL/egl.h>
    void egl_init();
#else
    GLFWwindow* glfw_init(const int width=1280, const int height=800);

#endif

    // init some buffer objects(XBO) and texture objects(XTO)
    //

    GLuint InitPBO();

    GLuint InitSSBO(int size);

}//namespace opengl
#endif
