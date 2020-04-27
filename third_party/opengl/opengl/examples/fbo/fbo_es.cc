#define GLFW_INCLUDE_ES31
#include <GLFW/glfw3.h>

#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>
#include <vector>
#include <random>

// global variables here
const int kWindowWidth = 640;
const int kWindowHeight = 480;
GLFWwindow* window_;
GLuint vertex_shader_;
GLuint fragment_shader;
GLuint program;

// Don't need to change this.
// We want to draw 2 giant triangles that cover the whole screen.
struct Vertex {
    float x, y;
};

// This is the main part.
static const char *fragment_shader_text = "#version 300 es\n"
"precision mediump float;\n"
"uniform sampler2D A;\n"
"uniform sampler2D B;\n"
"uniform int N;\n"
"out float color;\n"
"void main() {\n"
"  ivec2 pixel = ivec2(gl_FragCoord.xy);\n"
"  int idx = pixel.x;\n"
"  int row = idx / N;\n"
"  int col = idx % N;\n"
"  color = 0.0;\n"
"  for (int i = 0; i < N; i++) {\n"
"    float a = texelFetch(A, ivec2(row * N + i, 0), 0).r;\n"
"    float b = texelFetch(B, ivec2(i * N + col, 0), 0).r;\n"
"    color += a * b;\n"
"  }\n"
"}\n";



static constexpr size_t kNumVertices = 6;

const char *vertex_shader_text_ = "#version 300 es\n"
"in vec2 point; // input to vertex shader\n"
"void main() {\n"
"  gl_Position = vec4(point, 0.0, 1.0);\n"
"}\n";

const Vertex vertices[kNumVertices] = {
    {-1.f, -1.f},
    {1.0f, -1.f},
    {1.0f, 1.0f},
    {-1.f, -1.f},
    {-1.f, 1.0f},
    {1.0f, 1.0f},
};

inline const char *GLGetErrorString(GLenum error) {
    switch (error) {
        case GL_NO_ERROR:
            return "GL_NO_ERROR";
        case GL_INVALID_ENUM:
            return "GL_INVALID_ENUM";
        case GL_INVALID_VALUE:
            return "GL_INVALID_VALUE";
        case GL_INVALID_OPERATION:
            return "GL_INVALID_OPERATION";
            //case GL_STACK_OVERFLOW:
            // return "GL_STACK_OVERFLOW";
            //case GL_STACK_UNDERFLOW:
            // return "GL_STACK_UNDERFLOW";
        case GL_OUT_OF_MEMORY:
            return "GL_OUT_OF_MEMORY";
        default:
            return "Unknown OpenGL error code";
    }
}

void OPENGL_CHECK_ERROR(int line) {
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "OpenGL error, code=" << err << ": "
            << GLGetErrorString(err)<<"\nin main.cpp: "<<line << std::endl;
        assert(false);
    }
}

/*!
 * \brief Protected OpenGL call.
 * \param func Expression to call.
 */
#define OPENGL_CALL(func)                                                      \
{                                                                            \
    (func);                                                                    \
    OPENGL_CHECK_ERROR(__LINE__);                                                      \
}

void GlfwErrorCallback(int err, const char *str) {
    std::cerr << "Error: [" << err << "] " << str << std::endl;
}

/*!
 * \brief Create and compile a shader from a source string.
 * \param shader_kind The kind of shader.
 * Could be GL_VERTEX_SHADER or GL_FRAGMENT_SHADER.
 * \param shader_src The source string of the shader.
 * \return The compiled shader ID.
 */
GLuint CreateShader(GLenum shader_kind, const char *shader_src) {
    // Create the shader.
    GLuint shader = glCreateShader(shader_kind);
    glShaderSource(shader, 1, &shader_src, nullptr);
    glCompileShader(shader);

    // Check compile errors.
    GLint err;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &err);

    GLint info_log_len;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &info_log_len);

    if (info_log_len > 0) {
        std::unique_ptr<char[]> err_msg(new char[info_log_len + 1]);
        glGetShaderInfoLog(shader, info_log_len, nullptr, err_msg.get());
        std::cout << err_msg.get() << std::endl;
        assert(false);
    }

    OPENGL_CHECK_ERROR(__LINE__);

    return shader;
}

void glfw_init(){
    // Set an error handler.
    // This can be called before glfwInit().
    glfwSetErrorCallback(&GlfwErrorCallback);

    // Initialize GLFW.
    if (glfwInit() != GL_TRUE) {
        std::cout << "glfwInit() failed!" << std::endl;
        assert(false);
    }

    // Create a window.
    // TODO(zhixunt): GLFW allows us to create an invisible window.
    // TODO(zhixunt): On retina display, window size is different from framebuffer size.
    glfwWindowHint(GLFW_CLIENT_API,  GLFW_OPENGL_ES_API);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_VISIBLE, GL_FALSE);
    //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    //glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    window_ = glfwCreateWindow(kWindowWidth, kWindowHeight, "", nullptr, nullptr);
    if (window_ == nullptr) {
        std::cout << "glfwCreateWindow() failed!" << std::endl;
        assert(false);
    }

    std::cout << "GLFW says OpenGL version: "
        << glfwGetWindowAttrib(window_, GLFW_CONTEXT_VERSION_MAJOR)
        << "."
        << glfwGetWindowAttrib(window_, GLFW_CONTEXT_VERSION_MINOR)
        << "."
        << glfwGetWindowAttrib(window_, GLFW_CONTEXT_REVISION)
        << std::endl;

    // Before using any OpenGL API, we must specify a context.
    glfwMakeContextCurrent(window_);

    // Must be called after creating GLFW window.
    //glewInit();

    std::cout << "Opengl says version: " << glGetString(GL_VERSION) << std::endl;

    OPENGL_CHECK_ERROR(__LINE__);

    // We always render the same vertices and triangles.
    GLuint vertex_buffer;
    OPENGL_CALL(glGenBuffers(1, &vertex_buffer));
    OPENGL_CALL(glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer));
    OPENGL_CALL(glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices,
                GL_STATIC_DRAW));

    GLuint vertex_array;
    OPENGL_CALL(glGenVertexArrays(1, &vertex_array));
    OPENGL_CALL(glBindVertexArray(vertex_array));
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);

    // We always use the same vertex shader.
    vertex_shader_ = CreateShader(GL_VERTEX_SHADER, vertex_shader_text_);
}

void glfw_destory(){
    // Paired with glfwCreateWindow().
    glfwDestroyWindow(window_);

    // Paired with glfwInit().
    glfwTerminate();
}

/*!
 * \brief Create a program that uses the given vertex and fragment shaders.
 * \param fragment_shader The **compiled** fragment shader.
 * \return The program ID.
 */
void CreateProgram() {
    // Create the program and link the shaders.
    program = glCreateProgram();
    glAttachShader(program, vertex_shader_);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);

    // Check link errors.
    GLint err;
    glGetProgramiv(program, GL_LINK_STATUS, &err);

    GLint info_log_len;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &info_log_len);

    if (info_log_len > 0) {
        std::unique_ptr<char[]> err_msg(new char[info_log_len + 1]);
        glGetProgramInfoLog(program, info_log_len, nullptr, err_msg.get());
        std::cout << err_msg.get() << std::endl;
        assert(false);
    }

    OPENGL_CHECK_ERROR(__LINE__);

    OPENGL_CALL(glDetachShader(program, vertex_shader_));
    OPENGL_CALL(glDetachShader(program, fragment_shader));

    auto point_attrib = GLuint(glGetAttribLocation(program, "point"));
    OPENGL_CALL(glEnableVertexAttribArray(point_attrib));

    OPENGL_CALL(glVertexAttribPointer(point_attrib, 2, GL_FLOAT, GL_FALSE,
                sizeof(Vertex), nullptr));
}


GLuint CreateTexture(const GLfloat *data, GLsizei width_, GLsizei height_){
    GLuint texture_;

    // Create a texture.
    OPENGL_CALL(glGenTextures(1, &texture_));

    std::clog << "Created texture [" << texture_ << "]" << std::endl;

    // Bind to temporary unit.
    // workspace.BindTextureUnit(workspace.NumTextureUnits() - 1, texture_);
    glBindTexture(GL_TEXTURE_2D, texture_);

    // Similar to cudaMemcpy.
    OPENGL_CALL(glTexImage2D(GL_TEXTURE_2D, /*level=*/0, GL_RGBA32F,
                width_, height_, /*border=*/0,
                GL_RGBA, GL_FLOAT, data));
    // TODO(zhixunt): What are these?
    OPENGL_CALL(
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    OPENGL_CALL(
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
    OPENGL_CALL(
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
    OPENGL_CALL(
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
    return texture_;
}

void SetInput( std::string name, GLuint id,  int tex_id){
    GLint location= glGetUniformLocation(program, name.c_str());
    glUniform1i(location, tex_id);
    glActiveTexture(GL_TEXTURE0+tex_id);
    glBindTexture(GL_TEXTURE_2D, id);
}

void Download(GLfloat *data, GLint width, GLint height, GLuint texture,int tex_id){
    OPENGL_CALL(glActiveTexture(GL_TEXTURE0 + tex_id));
    OPENGL_CALL(glBindTexture(GL_TEXTURE_2D, texture));
    glReadPixels(0, 0, width, height, GL_RGBA, GL_FLOAT, data);
}

// void Render(GLuint Program, ){
// }

int main(){
    const int N = 10;
    GLint width = N;
    GLint height = N;
    const int niters = 10;
    const size_t texture_size = width*height*4;
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> dist(1.0f, 2.0f);

    glfw_init();
    // create fragment shader and link it with vertex
    // shader to create program
    fragment_shader = CreateShader(GL_FRAGMENT_SHADER,
            fragment_shader_text);
    CreateProgram();

    //////////////////////////////////////////////////////
    // prepare data
    //////////////////////////////////////////////////////
    std::vector<GLfloat> texture0_data(texture_size, 0.25f);
    std::vector<GLfloat> texture1_data(texture_size, 0.25f);
    for (size_t i = 0; i != texture_size; ++i) {
        texture0_data[i] = dist(mt);
        texture1_data[i] = dist(mt);
    }
    auto input0 = CreateTexture(texture0_data.data(), width, height);
    auto input1 = CreateTexture(texture1_data.data(), width, height);

    auto output = CreateTexture(nullptr, width, height);

    ////////////////////////////////////////
    // Compute(Render)
    ////////////////////////////////////////
    OPENGL_CALL(glUseProgram(program));

    // Create frame buffer And Check its Completation
    GLuint frame_buffer;
    OPENGL_CALL(glGenFramebuffers(1, &frame_buffer));
    OPENGL_CALL(glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer));
    // Set "renderedTexture" as our colour attachement #0
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,
            output , 0);

    // Set the list of draw buffers.
    // GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
    // "1" is the size of DrawBuffers.
    // OPENGL_CALL(glDrawBuffers(1, DrawBuffers));

    // Always check that our framebuffer is ok
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cout << "Framebuffer not complete." << std::endl;
        assert(false);
    }

    // Tell the fragment shader what input textures to use.
    SetInput("A", input0, 0);
    SetInput("B", input1, 1);
    OPENGL_CALL(glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer));
    OPENGL_CALL(glViewport(0, 0, width, height));

    auto opengl_start = std::chrono::system_clock::now();
    for (int iter = 0; iter < niters; ++iter) {
        OPENGL_CALL(glClear(GL_COLOR_BUFFER_BIT));
        OPENGL_CALL(glDrawArrays(GL_TRIANGLES, 0, 6));
        glFinish();
    }

    glDeleteFramebuffers(1, &frame_buffer);

    auto opengl_end = std::chrono::system_clock::now();
    std::cout << "opengl: "
        << ((opengl_end - opengl_start).count() / niters)
        << std::endl;

    ///////////////////////////////////////
    // Download output Data from Device
    std::vector<GLfloat> retrieved_data(static_cast<size_t>(width * height));
    // Download(retrieved_data.data(), width, height, output, );
    // for (GLuint unit = 0; unit != inputs.size(); ++unit) {
    // const std::string &name = inputs[unit].first;
    // Texture *texture = inputs[unit].second;

    // BindTextureUnit(unit, *texture);

    // GLint texture_uniform = glGetUniformLocation(program.program_,
    // name.c_str());
    // OPENGL_CALL(glUniform1i(texture_uniform, unit));
    // }


    // Always check that our framebuffer is ok
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cout << "Framebuffer not complete." << std::endl;
        assert(false);
    }

    std::vector<GLfloat> cpu_result(static_cast<size_t>(texture_size));
    auto cpu_start = std::chrono::system_clock::now();
    // for (int iter = 0; iter < niters; ++iter) {
    // for (int row = 0; row != N; ++row) {
    // for (int col = 0; col != N; ++col) {
    // cpu_result[row * N + col] = 0.0f;
    // for (int i = 0; i != N; ++i) {
    // GLfloat a = texture0_data[row * N + i];
    // GLfloat b = texture1_data[i * N + col];
    // cpu_result[row * N + col] += a * b;
    // }
    // }
    // }
    // }
    auto cpu_end = std::chrono::system_clock::now();

    // for (int i = 0; i < retrieved_data.size(); ++i) {
    // assert(std::abs(retrieved_data[i] - cpu_result[i]) < 0.001f);
    // }

    std::cout << "cpu:    "
        << ((cpu_end - cpu_start).count() / niters)
        << std::endl;

    glfw_destory();
    return 0;
}

