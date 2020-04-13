#ifndef MACROS_H_
#define MACROS_H_
#include "opengl/core/opengl.h"

// #ifdef OPEN_GL_CHECK_ERROR
#define OPENGL_CHECK_ERROR              \
{                                   \
    GLenum error = glGetError();    \
    if (GL_NO_ERROR != error){       \
        LOG(FATAL)<<"error here"; \
    }\
}
// #else
// #define OPENGL_CHECK_ERROR
// #endif

#endif
