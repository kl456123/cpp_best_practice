#include <iostream>
#include <cmath>
#include <fstream>
#include <string>

#define GLWE_STATIC
#include <GL/glew.h>
#include <GL/gl.h>
#include <GLFW/glfw3.h>


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
        // return EXIT_FAILURE;
    }

    // Create Context and Load OpenGL Functions
    glfwMakeContextCurrent(window);
    return window;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height){
    glViewport(0, 0, width, height);
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

std::string load_source(const std::string& fname){
    std::ifstream fd(fname);
    auto src = std::string(std::istreambuf_iterator<char>(fd),
            (std::istreambuf_iterator<char>()));
    fd.close();
    return src;
}

void processInput(GLFWwindow* window){
    if(glfwGetKey(window, GLFW_KEY_ESCAPE)==GLFW_PRESS){
        glfwSetWindowShouldClose(window, true);
    }
}

int main(int argc, char* argv[]){

    // init window and context
    auto window = glfw_init(1280, 800);

    // init glew
    glew_init();

    // register callback
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);


    // vetex shader

    auto vertexShaderSource = load_source("../demo/glsl/vertex.glsl");
    const char* vertex_src = vertexShaderSource.c_str();

    GLuint  vertexShader;
    vertexShader = glCreateShader(GL_VERTEX_SHADER);

    glShaderSource(vertexShader, 1, &vertex_src, NULL);
    // compile
    glCompileShader(vertexShader);

    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);

    if(!success){
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cout<<"ERROR::SHADER:VERTEX::COMPILATION_FAILED\n"<<infoLog<<std::endl;
    }

    auto fragmentShaderSource = load_source("../demo/glsl/fragment.glsl");
    const char* frag_src = fragmentShaderSource.c_str();
    GLuint fragmentShader;
    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &frag_src, NULL);
    glCompileShader(fragmentShader);

    // program
    GLuint shaderProgram;
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    // check error if any
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if(!success){
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cout<<"ERROR::PROGRAM::LINK_FAILED\n"<<infoLog<<std::endl;
    }


    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // VAO
    GLuint vao;
    glGenVertexArrays(1, &vao);
    // bind it first
    glBindVertexArray(vao);

    // mem in cpu
    // triangle
    // float vertices[] = {
    // -0.5f, -0.5f, 0.0f,
    // 0.5f, -0.5f, 0.0f,
    // 0.0f, 0.5f, 0.0f
    // };

    // rectangle
    // float vertices[] = {
        // 0.5f, 0.5f, 0.0f,
        // 0.5f, -0.5f, 0.0f,
        // -0.5f, -0.5f, 0.0f,
        // -0.5f, 0.5f, 0.0f
    // };
    // rectangle with colors
    float vertices[] = {
        // positions            // colors
        0.5f, -0.5f, 0.0f,      1.0f, 0.0f, 0.0f,
        -0.5f, -0.5f, 0.0f,     0.0f, 1.0f, 0.0f,
        0.0f, 0.5f, 0.0f,       0.0f, 0.0f, 1.0f
    };

    GLuint VBO;
    // create new name(id)
    glGenBuffers(1, &VBO);

    // bind
    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    // allocate memory in gpu
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // set attrs(inputs for shader)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6*sizeof(float), 0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6*sizeof(float), (void*)(3*sizeof(float)));
    glEnableVertexAttribArray(1);

    // EBO

    unsigned int indices[] = {0,1,3,1,2,3};
    GLuint EBO;
    glGenBuffers(1, &EBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    // poll
    while(!glfwWindowShouldClose(window)){
        // keyboard input
        processInput(window);

        // render here

        // clear color first
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);// set state
        glClear(GL_COLOR_BUFFER_BIT);// use state

        glUseProgram(shaderProgram);
        glBindVertexArray(vao);

        // set uniform
        // float timeValue = glfwGetTime();
        // float greenValue = (sin(timeValue)/2.0f)+0.5f;
        // int vertexColorLocation = glGetUniformLocation(shaderProgram, "outColor");
        // // after use program
        // glUniform4f(vertexColorLocation, 0.0f, greenValue, 0.0f, 1.0f);

        glDrawArrays(GL_TRIANGLES, 0, 3);
        // glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // clean up

    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &VBO);

    glfwTerminate();

    return 0;
}
