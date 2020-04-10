#include <iostream>
#include <cmath>
#include <fstream>
#include <string>

// opengl
#define GLWE_STATIC
#include <GL/glew.h>
#include <GL/gl.h>
#include <GLFW/glfw3.h>

// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// glm
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

glm::mat4 get_transform_3d(bool rotate);
glm::mat4 get_transform_3d(glm::mat4 model);

glm::vec3* get_location(int* num_objects){
    static glm::vec3 cubePositions[] = {
        glm::vec3( 0.0f,  0.0f,  0.0f),
        glm::vec3( 2.0f,  5.0f, -15.0f),
        glm::vec3(-1.5f, -2.2f, -2.5f),
        glm::vec3(-3.8f, -2.0f, -12.3f),
        glm::vec3( 2.4f, -0.4f, -3.5f),
        glm::vec3(-1.7f,  3.0f, -7.5f),
        glm::vec3( 1.3f, -2.0f, -2.5f),
        glm::vec3( 1.5f,  2.0f, -2.5f),
        glm::vec3( 1.5f,  0.2f, -1.5f),
        glm::vec3(-1.3f,  1.0f, -1.5f)
    };
    *num_objects = sizeof(cubePositions)/sizeof(cubePositions[0]);
    return cubePositions;
}

glm::mat4 get_transform(){
    glm::vec4 vec(1.0f, 0.0f, 0.0f, 1.0f);
    glm::mat4 trans = glm::mat4(1.0f);
    trans = glm::translate(trans, glm::vec3(1.0f, 1.0f, 0.0f));
    vec = trans * vec;
    std::cout<< vec.x<<vec.y<<vec.z<<std::endl;

    // rotate
    trans = glm::mat4(1.0f);
    trans = glm::rotate(trans, glm::radians(90.0f), glm::vec3(0.0, 0.0, 1.0));
    trans = glm::scale(trans, glm::vec3(0.5, 0.5, 0.5));

    return trans;
}

glm::mat4 get_transform_3d(bool rotate=false){
    // model
    glm::mat4 model = glm::mat4(1.0f);
    if(rotate){
        model = glm::rotate(model, (float)glfwGetTime()* glm::radians(50.0f),
                glm::vec3(0.5f, 1.0f, 0.0f));
    }else{
        model=glm::rotate(model, glm::radians(-55.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    }

    return get_transform_3d(model);
}

glm::mat4 get_transform_3d(glm::mat4 model){

    // traslation
    glm::mat4 view = glm::mat4(1.0f);
    view = glm::translate(view, glm::vec3(0.0f, 0.0f, -3.0f));

    // projection
    glm::mat4 projection;
    projection = glm::perspective(glm::radians(45.0f), 800.0f/600.0f, 0.1f, 100.0f);

    return projection* view * model;
}

glm::mat4 get_transform_animation(){
    glm::mat4 trans = glm::mat4(1.0f);
    trans = glm::translate(trans, glm::vec3(0.5f, -0.5f, 0.0f));
    trans = glm::rotate(trans, (float)glfwGetTime(), glm::vec3(0.0f, 0.0f, 1.0f));
    return trans;
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
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5*sizeof(float), 0);
    glEnableVertexAttribArray(0);

    // color
    // glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8*sizeof(float), (void*)(3*sizeof(float)));
    // glEnableVertexAttribArray(1);

    // texture
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)(3*sizeof(float)));
    glEnableVertexAttribArray(1);
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
    // static float vertices[] = {
    // // positions          // colors           // texture coords
    // 0.5f,  0.5f, 0.0f,   1.0f, 0.0f, 0.0f,   1.0f, 1.0f,   // top right
    // 0.5f, -0.5f, 0.0f,   0.0f, 1.0f, 0.0f,   1.0f, 0.0f,   // bottom right
    // -0.5f, -0.5f, 0.0f,   0.0f, 0.0f, 1.0f,   0.0f, 0.0f,   // bottom left
    // -0.5f,  0.5f, 0.0f,   1.0f, 1.0f, 0.0f,   0.0f, 1.0f    // top left
    // };

    static float vertices[] = {
        -0.5f, -0.5f, -0.5f,  0.0f, 0.0f,
        0.5f, -0.5f, -0.5f,  1.0f, 0.0f,
        0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
        0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
        -0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 0.0f,

        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
        0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
        0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
        0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
        -0.5f,  0.5f,  0.5f,  0.0f, 1.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,

        -0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
        -0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
        -0.5f,  0.5f,  0.5f,  1.0f, 0.0f,

        0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
        0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
        0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
        0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
        0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
        0.5f,  0.5f,  0.5f,  1.0f, 0.0f,

        -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
        0.5f, -0.5f, -0.5f,  1.0f, 1.0f,
        0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
        0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,

        -0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
        0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
        0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
        0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
        -0.5f,  0.5f,  0.5f,  0.0f, 0.0f,
        -0.5f,  0.5f, -0.5f,  0.0f, 1.0f
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

void SetMatrix(GLuint program, const std::string& name, glm::mat4& trans){
    glUseProgram(program);
    int vertexColorLocation = glGetUniformLocation(program, name.c_str());
    glUniformMatrix4fv(vertexColorLocation, 1, GL_FALSE, glm::value_ptr(trans));
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

    int num_objects=0;
    glm::vec3* cubePositions = get_location(&num_objects);
    glEnable(GL_DEPTH_TEST);

    // poll
    while(!glfwWindowShouldClose(window)){
        // keyboard input
        processInput(window);
        // render here

        // clear color first
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);// set state
        glClear(GL_COLOR_BUFFER_BIT);// use state

        // clear z-buffer
        glClear(GL_COLOR_BUFFER_BIT |GL_DEPTH_BUFFER_BIT);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture1);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, texture2);

        // glm::mat4 trans = get_transform_3d(true);
        // SetMatrix(shaderProgram, "transform", trans);

        glUseProgram(shaderProgram);

        glBindVertexArray(VAO);
        // glDrawArrays(GL_TRIANGLES, 0, 36);
        // glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        for(int i=0;i<num_objects;++i){
            glm::mat4 model = glm::mat4(1.0f);
            model = glm::translate(model, cubePositions[i]);
            float angle = 20.0f * i;
            model = glm::rotate(model, (float)glfwGetTime() * glm::radians(angle),
                    glm::vec3(1.0f, 0.3f, 0.5f));
            glm::mat4 trans = get_transform_3d(model);
            SetMatrix(shaderProgram, "transform", trans);
            glDrawArrays(GL_TRIANGLES, 0, 36);
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // clean up
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);

    glfwTerminate();

    return 0;
}
