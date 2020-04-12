#include <stdio.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <string.h>

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
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
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

int glew_init(){
    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if(err!=GLEW_OK){
        std::cout<<"glewInit failed: "<<glewGetErrorString(err)<<std::endl;
        return -1;
    }
    return 0;
}

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

void SetInt(GLuint program, const std::string& name, GLuint texture_id){
    glUseProgram(program);
    glUniform1i(glGetUniformLocation(program, name.c_str()), texture_id);
}

GLuint InitSSBO(int size){
    GLuint SSBO;
    glGenBuffers(1, &SSBO);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, SSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER, size, NULL, GL_DYNAMIC_DRAW);
    return SSBO;
}

int main(){
    // init window and context
    auto window = glfw_init(1280, 800);

    // init glew
    glew_init();

    // Creating the Texture / Image
    // dimensions of the image
    // just use single channel(R in RGBA)
    // shape
    typedef float DataType;
    int tex_w = 512, tex_h = 512;
    GLenum internal_format= GL_RGBA32F;
    GLenum format = GL_RGBA;
    GLenum type = GL_FLOAT;
    const int channels = 4;
    std::vector<int> image_shape({tex_h, tex_w, channels});
    // num of elements
    int num = 1;
    for(auto dim:image_shape){
        num*=dim;
    }
    // size
    const int size = num * sizeof(DataType);

    GLuint tex_output;
    glGenTextures(1, &tex_output);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex_output);
    // wrapping and filter
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

    DataType* image_data = new DataType[num];
    for(int i=0;i<num;i++){
        image_data[i] = random()%256;
    }

    // storage
    // glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, tex_w, tex_h, 0, GL_R,
            // GL_FLOAT, image_data);

    glTexImage2D(GL_TEXTURE_2D, 0, internal_format, tex_w, tex_h,
            0, format, type, image_data);


    // glBindImageTexture(0, tex_output, 0, GL_TRUE, 0, GL_READ_ONLY, GL_R32F);
    // work group size
    int work_grp_cnt[3];
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &work_grp_cnt[0]);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, &work_grp_cnt[1]);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, &work_grp_cnt[2]);
    printf("max global (total) work group counts x:%i y:%i z:%i\n",
            work_grp_cnt[0], work_grp_cnt[1], work_grp_cnt[2]);

    int work_grp_inv;
    glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &work_grp_inv);

    // read glsl
    std::ifstream ifs("../gpgpu/compute_shader.glsl");
    std::string content( (std::istreambuf_iterator<char>(ifs) ),
            (std::istreambuf_iterator<char>()));
    if(content.empty()){
        std::cout<<"Read File ERROR"<<std::endl;
        return -1;
    }
    const char* the_ray_shader_string = content.c_str();

    // compute shader
    GLuint ray_shader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(ray_shader, 1, &the_ray_shader_string, NULL);
    glCompileShader(ray_shader);

    int success;
    char infoLog[512];
    glGetShaderiv(ray_shader, GL_COMPILE_STATUS, &success);

    if(!success){
        glGetShaderInfoLog(ray_shader, 512, NULL, infoLog);
        std::cout<<"ERROR::SHADER:VERTEX::COMPILATION_FAILED\n"<<infoLog<<std::endl;
    }

    GLuint ray_program = glCreateProgram();
    glAttachShader(ray_program, ray_shader);
    glLinkProgram(ray_program);

    // check error if any
    glGetProgramiv(ray_program, GL_LINK_STATUS, &success);
    if(!success){
        glGetProgramInfoLog(ray_program, 512, NULL, infoLog);
        std::cout<<"ERROR::PROGRAM::LINK_FAILED\n"<<infoLog<<std::endl;
    }


    // SetInt(ray_program, "image_input", 0);


    GLuint SSBO = InitSSBO(size);
    DataType *buffer_cpu = new DataType[num];
    memset(buffer_cpu, 0, size);

    //dispatch the shaders
    {
        glUseProgram(ray_program);
        // input image
        // glBindImageTexture(0, tex_output, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F);
        // glActiveTexture(GL_TEXTURE0);
        // glUniform1i(0, 0);
        // glBindTexture(GL_TEXTURE_2D, tex_output);
        glActiveTexture(GL_TEXTURE0);
        glUniform1i(0, 0);
        glBindTexture(GL_TEXTURE_2D, tex_output);

        glUniform2i(2, image_shape[0], image_shape[1]);

        // output
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, SSBO);
        glDispatchCompute((GLuint)tex_w, (GLuint)tex_h, 1);
    }

    // sync
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    {
        // map back to cpu
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, SSBO);
        auto ptr = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, size, GL_MAP_WRITE_BIT);
        ::memcpy(buffer_cpu, ptr, size);
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    }

    // just print
    for(int i=0;i<100;i++){
        std::cout<<(DataType)image_data[i]<<" ";
    }

    std::cout<<std::endl;
    std::cout<<std::endl;
    for(int i=0;i<100;i++){
        std::cout<<(DataType)buffer_cpu[i]<<" ";
    }
    std::cout<<std::endl;


    // while(!glfwWindowShouldClose(window)){


    // {
    // // glClear(GL_COLOR_BUFFER_BIT);
    // // glUseProgram(quad_program);
    // }

    // glfwPollEvents();
    // if (GLFW_PRESS == glfwGetKey(window, GLFW_KEY_ESCAPE)) {
    // glfwSetWindowShouldClose(window, 1);
    // }
    // glfwSwapBuffers(window);
    // }


    return 0;
}
