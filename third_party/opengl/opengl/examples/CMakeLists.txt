
option(BUILD_RENDER "BUILD OPENGL RENDER" ON)
option(BUILD_GPGPU "BUILD OPENGL GPGPU" ON)
option(BUILD_GLES "BUILD GLES DEMO" ON)
option(BUILD_GLUT "BUILD GLUT DEMO" ON)


if(BUILD_RENDER)
    add_executable(render ${CMAKE_CURRENT_SOURCE_DIR}/render/render.cc)
    if(BUILD_ARM)
        target_link_libraries(render -lglfw -lGL ${OpenCV_LIBS})
    else()
        target_link_libraries(render -lglfw -lGLESv2 ${OpenCV_LIBS})
    endif()
endif()

if(BUILD_GPGPU)
    if(BUILD_ARM)
        set(AXPY_DEMO_OpenGL_LIB  -lglut -lGLEW -lGLESv2)
    else()
        set(AXPY_DEMO_OpenGL_LIB  -lglut -lGLEW -lGL -lGLU)
    endif()

    # this module use glut and glew, so we don't use OpenGL_LIB
    add_executable(axpy_demo ${CMAKE_CURRENT_SOURCE_DIR}/gpgpu/axpy_demo.cc)
    target_link_libraries(axpy_demo ${OpenCV_LIBS} ${AXPY_DEMO_OpenGL_LIB})

    add_executable(mrt_demo ${CMAKE_CURRENT_SOURCE_DIR}/gpgpu/mrt_demo.cc)
    target_link_libraries(mrt_demo ${OpenCV_LIBS} ${AXPY_DEMO_OpenGL_LIB})

    add_executable(compute_shader ${CMAKE_CURRENT_SOURCE_DIR}/gpgpu/compute_shader.cc)
    target_link_libraries(compute_shader ${OpenGL_LIB} ${OpenCV_LIBS} ${GL_CORE})
endif()

if(BUILD_GLES)
    # use glfw
    add_executable(glfw_gles_demo ${CMAKE_CURRENT_SOURCE_DIR}/gles/glfw.cc)
    target_link_libraries(glfw_gles_demo -lGLESv2 -lglfw)

    # use egl
    add_executable(egl_gles_demo ${CMAKE_CURRENT_SOURCE_DIR}/gles/egl.cc)
    target_link_libraries(egl_gles_demo -lGLESv2 -lEGL glog::glog)
endif()

if(BUILD_GLUT)
    add_executable(glut_demo ${CMAKE_CURRENT_SOURCE_DIR}/glut/glut_demo.cc)
    target_link_libraries(glut_demo -lglut)
endif()
