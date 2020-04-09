#include <iostream>
#include <cmath>
#include <fstream>
#include <string>

#define GLWE_STATIC
#include <GL/glew.h>
#include <GL/gl.h>
#include <GLFW/glfw3.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


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

enum ShaderType{
    VERTEX,
    FRAGMENT
};

GLuint CreateShader(const std::string fname, GLenum type){

    auto vertexShaderSource = load_source(fname);
    const char* vertex_src = vertexShaderSource.c_str();

    GLuint  vertexShader;
    vertexShader = glCreateShader(type);

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
    return vertexShader;
}

GLuint CreateProgram(std::vector<GLuint> shaders){
    // program
    int success;
    char infoLog[512];

    GLuint shaderProgram;
    shaderProgram = glCreateProgram();
    for(auto& shader :shaders){
        glAttachShader(shaderProgram, shader);
    }
    glLinkProgram(shaderProgram);

    // check error if any
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if(!success){
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cout<<"ERROR::PROGRAM::LINK_FAILED\n"<<infoLog<<std::endl;
    }

    for(auto& shader :shaders){
        glDeleteShader(shader);
    }

    return shaderProgram;
}

GLuint InitVAO(){
    // VAO
    GLuint vao;
    glGenVertexArrays(1, &vao);
    // bind it first
    glBindVertexArray(vao);
    return vao;
}

GLuint InitVBO(const float* vertices, GLsizeiptr size){
    GLuint VBO;
    // create new name(id)
    glGenBuffers(1, &VBO);

    // bind
    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    // allocate memory in gpu
    glBufferData(GL_ARRAY_BUFFER, size, vertices, GL_STATIC_DRAW);

    return VBO;
}

void InitAttr(){
    // set attrs(inputs for shader)
    // position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8*sizeof(float), 0);
    glEnableVertexAttribArray(0);

    // color
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)(3*sizeof(float)));
    glEnableVertexAttribArray(1);

    // texture
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)(6*sizeof(float)));
    glEnableVertexAttribArray(2);
}

const float* MockData(int* size){
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
    // static float vertices[] = {
    // // positions            // colors
    // 0.5f, -0.5f, 0.0f,      1.0f, 0.0f, 0.0f,
    // -0.5f, -0.5f, 0.0f,     0.0f, 1.0f, 0.0f,
    // 0.0f, 0.5f, 0.0f,       0.0f, 0.0f, 1.0f
    // };
    static float vertices[] = {
        // positions          // colors           // texture coords
        0.5f,  0.5f, 0.0f,   1.0f, 0.0f, 0.0f,   1.0f, 1.0f,   // top right
        0.5f, -0.5f, 0.0f,   0.0f, 1.0f, 0.0f,   1.0f, 0.0f,   // bottom right
        -0.5f, -0.5f, 0.0f,   0.0f, 0.0f, 1.0f,   0.0f, 0.0f,   // bottom left
        -0.5f,  0.5f, 0.0f,   1.0f, 1.0f, 0.0f,   0.0f, 1.0f    // top left
    };

    *size = sizeof(vertices);

    return vertices;
}

GLuint InitEBO(){
    // EBO
    unsigned int indices[] = {0,1,3,1,2,3};
    GLuint EBO;
    glGenBuffers(1, &EBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
    return EBO;
}

void SetFloat(GLuint program, const std::string& name, float value[4]){
    glUseProgram(program);
    // set uniform
    int vertexColorLocation = glGetUniformLocation(program, name.c_str());
    // after use program
    glUniform4f(vertexColorLocation, value[0], value[1], value[2], value[3]);
}

void SetInt(GLuint program, const std::string& name, GLuint texture_id){
    glUseProgram(program);
    glUniform1i(glGetUniformLocation(program, name.c_str()), texture_id);
}

GLuint InitTexture(const std::string& image_name){
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    // set options(wrapping/filtering)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    auto image_orig = cv::imread(image_name.c_str());
    // some preprocesses
    cv::cvtColor(image_orig, image_orig, CV_BGR2RGB);
    cv::Mat image;
    cv::flip(image_orig, image, 0);
    int width = image.cols;
    int height = image.rows;
    unsigned char* data = image.data;

    if(data==nullptr){
        std::cout<<"Failed to load texture"<<std::endl;
        return 0;
    }

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);

    return texture;
}

int main(int argc, char* argv[]){

    // init window and context
    auto window = glfw_init(1280, 800);

    // init glew
    glew_init();

    // register callback
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // vetex shader
    GLuint vertexShader = CreateShader("../demo/glsl/vertex.glsl", GL_VERTEX_SHADER);

    GLuint fragmentShader = CreateShader("../demo/glsl/fragment.glsl", GL_FRAGMENT_SHADER);

    std::vector<GLuint> shaders({vertexShader, fragmentShader});
    GLuint shaderProgram = CreateProgram(shaders);


    int size=0;
    const float* vertices = MockData(&size);

    GLuint VAO = InitVAO();
    GLuint VBO = InitVBO(vertices, size);

    GLuint EBO = InitEBO();
    GLuint texture1 = InitTexture("../assets/container.jpg");
    GLuint texture2 = InitTexture("../assets/awesomeface.png");

    InitAttr();

    SetInt(shaderProgram, "texture1", 0);
    SetInt(shaderProgram, "texture2", 1);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    // poll
    while(!glfwWindowShouldClose(window)){
        // keyboard input
        processInput(window);
        // render here

        // clear color first
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);// set state
        glClear(GL_COLOR_BUFFER_BIT);// use state

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture1);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, texture2);

        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);

        // glDrawArrays(GL_TRIANGLES, 0, 3);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // clean up
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);

    glfwTerminate();

    return 0;
}
