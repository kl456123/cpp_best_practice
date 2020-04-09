#ifndef INIT_H_
#define INIT_H_
#include "opengl.h"

int glut_init(int argc, char* argv[]);
int glew_init();
GLFWwindow* glfw_init(const int width=1280, const int height=800);

#endif
