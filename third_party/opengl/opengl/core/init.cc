#include <iostream>

#include "init.h"



int glut_init(int argc, char* argv[]){
    glutInit(&argc, argv);

    int window = glutCreateWindow(argv[0]);
    return window;
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
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
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


// buffer and texture
GLuint InitPBO(){
    GLuint PBO;
    // create new name(id)
    glGenBuffers(1, &PBO);

    // bind
    glBindBuffer(GL_PIXEL_PACK_BUFFER, PBO);

    // allocate memory in gpu
    // glBufferData(GL_ARRAY_BUFFER, size, vertices, GL_STATIC_DRAW);

    return PBO;
}

GLuint InitSSBO(int size){
    GLuint SSBO;
    glGenBuffers(1, &SSBO);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, SSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, size, NULL, GL_DYNAMIC_DRAW);
    return SSBO;
}
