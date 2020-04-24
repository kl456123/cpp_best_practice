#ifndef OPENGL_H_
#define OPENGL_H_
/*********************************************************************
 * Here We just use glfw and glew to Init OpenGL API.
 * Of course You can also use glut to inplace of glfw and
 * use gloader to inplace of glew.
 * The important things is that we need glfw-like library to
 * use GUI api to operate windows, and use glew-like library to
 * load symbol from library according to the version of OpenGL
 * you use
 */

/////////////////////////////////
// headers for opengl, define
// all necessary macros
/////////////////////////////////



#ifndef ARM_PLATFORM
#define GL_GLEXT_PROTOTYPES
#define GLFW_INCLUDE_GLEXT
#else
#define GLFW_INCLUDE_ES31
#define GLFW_INCLUDE_GLEXT
#endif
#include <GLFW/glfw3.h>

// define default opengl version
#ifndef OPENGL_MAJOR_VERSION
#define OPENGL_MAJOR_VERSION 3
#endif

#ifndef OPENGL_MINOR_VERSION
#define OPENGL_MINOR_VERSION 3
#endif

#ifndef GLSL_VERSION
#define GLSL_VERSION 430
#endif


#endif
