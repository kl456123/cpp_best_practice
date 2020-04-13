
option(BUILD_RENDER "BUILD OPENGL RENDER" OFF)
option(BUILD_GPGPU "BUILD OPENGL GPGPU" ON)


if(BUILD_RENDER)
    add_executable(render ${CMAKE_CURRENT_SOURCE_DIR}/render/render.cc)
    target_link_libraries(render ${OpenGL_LIB} ${OpenCV_LIBS})
endif()

if(BUILD_GPGPU)
    add_executable(axpy_demo ${CMAKE_CURRENT_SOURCE_DIR}/gpgpu/axpy_demo.cc)
    target_link_libraries(axpy_demo ${OpenGL_LIB} ${OpenCV_LIBS} ${GL_CORE})

    add_executable(compute_shader ${CMAKE_CURRENT_SOURCE_DIR}/gpgpu/compute_shader.cc)
    target_link_libraries(compute_shader ${OpenGL_LIB} ${OpenCV_LIBS} ${GL_CORE})
endif()
