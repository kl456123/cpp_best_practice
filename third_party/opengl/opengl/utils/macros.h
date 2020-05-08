#ifndef OPENGL_UTILS_MACROS_H_
#define OPENGL_UTILS_MACROS_H_
#include <glog/logging.h>

#include "opengl/core/opengl.h"

namespace opengl{
    void OpenGLCheckErrorWithLocation(const char* fname, int line);
    const char *GLGetErrorString(GLenum error);
}//opengl

/*!
 * \brief Protected OpenGL call.
 * \param func Expression to call.
 */
#define OPENGL_CALL(func)                                                      \
{                                                                            \
    (func);                                                                    \
    ::opengl::OpenGLCheckErrorWithLocation(__FILE__, __LINE__);                  \
}

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


#define UP_DIV(x, y)   (((x) + (y) - (1)) / (y))

#endif
