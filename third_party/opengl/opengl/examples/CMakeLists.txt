
option(BUILD_RENDER "BUILD OPENGL RENDER" ON)
option(BUILD_GPGPU "BUILD OPENGL GPGPU" ON)
option(BUILD_GLES "BUILD GLES DEMO" ON)
option(BUILD_GLUT "BUILD GLUT DEMO" ON)


if(BUILD_RENDER AND (NOT BUILD_ARM))
    add_executable(render ${CMAKE_CURRENT_SOURCE_DIR}/render/render.cc)
    target_link_libraries(render ${OpenGL_LIB} ${OpenCV_LIBS})
endif()

if(BUILD_GPGPU)
    add_executable(axpy_demo ${CMAKE_CURRENT_SOURCE_DIR}/gpgpu/axpy_demo.cc)
    target_link_libraries(axpy_demo ${OpenGL_LIB} ${OpenCV_LIBS} ${GL_CORE} -lglut)

    add_executable(compute_shader ${CMAKE_CURRENT_SOURCE_DIR}/gpgpu/compute_shader.cc)
    target_link_libraries(compute_shader ${OpenGL_LIB} ${OpenCV_LIBS} ${GL_CORE})
endif()

if(BUILD_GLES)
    # use glfw
    add_executable(glfw_gles_demo ${CMAKE_CURRENT_SOURCE_DIR}/gles/glfw.cc)
    target_link_libraries(glfw_gles_demo -lGLESv2 -lglfw)

    # use egl
    add_executable(egl_gles_demo ${CMAKE_CURRENT_SOURCE_DIR}/gles/egl.cc)
    target_link_libraries(egl_gles_demo -lGLESv2 -lEGL glog::glog -lGLEW)
endif()

if(BUILD_GLUT)
    add_executable(glut_demo ${CMAKE_CURRENT_SOURCE_DIR}/glut/glut_demo.cc)
    target_link_libraries(glut_demo -lglut)
endif()
