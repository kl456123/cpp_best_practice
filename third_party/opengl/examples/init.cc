#include <iostream>

#include "init.h"



int glut_init(int argc, char* argv[]){
    glutInit(&argc, argv);

    int window = glutCreateWindow(argv[0]);
    return 0;
}

int glew_init(){
    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if(err!=GLEW_OK){
        std::cout<<"glewInit failed: "<<glewGetErrorString(err)<<std::endl;
        return -1;
    }
    return 0;
}


GLFWwindow* glfw_init(const int width, const int height){
    // Load GLFW and Create a Window
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
    auto window = glfwCreateWindow(width, height, "OpenGL", nullptr, nullptr);

    // Check for Valid Context
    if (window == nullptr) {
        fprintf(stderr, "Failed to Create OpenGL Context");
        return nullptr;
    }

    // Create Context and Load OpenGL Functions
    glfwMakeContextCurrent(window);
    return window;
}
