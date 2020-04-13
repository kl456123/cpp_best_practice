#ifndef OPENGL_UTILS_MACROS_H_
#define OPENGL_UTILS_MACROS_H_
#include "opengl/core/opengl.h"

#ifdef ENABLE_OPENGL_CHECK_ERROR
#define OPENGL_CHECK_ERROR              \
{                                   \
    GLenum error = glGetError();    \
    if (GL_NO_ERROR != error){       \
        LOG(FATAL)<<"error here"; \
    }\
}
#else
#define OPENGL_CHECK_ERROR
#endif

#define EXPECT_OPENGL_NO_ERROR  \
    EXPECT_TRUE(glGetError()==GL_NO_ERROR)

#define ASSERT_OPENGL_NO_ERROR  \
    ASSERT_TRUE(glGetError()==GL_NO_ERROR)

#endif
