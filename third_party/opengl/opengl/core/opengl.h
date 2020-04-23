#ifndef OPENGL_H_
#define OPENGL_H_

/////////////////////////////////
// headers for opengl, define
// all necessary macros
/////////////////////////////////

#define GL_GLEXT_PROTOTYPES
#define GLWE_STATIC
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/freeglut.h>

#include <GLFW/glfw3.h>

#ifdef ARM_PLATFORM
// #include <GLES2/gl2.h>
// #include <GLES2/gl2ext.h>
#endif


#endif
