#include <string.h>
#include <fstream>

#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>
#include <vector>
#include <random>

#include "opengl/examples/fbo/context.h"

GLuint vertex_shader_;
GLuint fragment_shader;
GLuint program;

// core parameters, maybe some data types cannot supported
// in some platform
// four channel version
// const int num_channels = 4;
// GLenum internal_format = GL_RGBA32F;
// GLenum format = GL_RGBA;

// single channel version
const int num_channels = 1;
GLenum internal_format = GL_R32F;
GLenum format = GL_RED;

// Don't need to change this.
// We want to draw 2 giant triangles that cover the whole screen.
struct Vertex {
    float x, y;
};


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
        LOG(FATAL) << err_msg.get();
    }

    OPENGL_CHECK_ERROR;

    return shader;
}

void CreateVertexShader(){
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


/*!
 * \brief Create a program that uses the given vertex and fragment shaders.
 * \param fragment_shader The **compiled** fragment shader.
 * \return The program ID.
 */
void CreateProgram(const std::string fname) {
    std::ifstream fd(fname);
    std::string src = std::string(std::istreambuf_iterator<char>(fd),
            (std::istreambuf_iterator<char>()));
    if(src.empty()){
        std::cerr<<"Read File ERROR from "<<fname;
    }

    // create fragment shader and link it with vertex
    // shader to create program
    fragment_shader = CreateShader(GL_FRAGMENT_SHADER,
            src.c_str());

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
        LOG(FATAL) << err_msg.get();
    }

    OPENGL_CHECK_ERROR;

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

    LOG(INFO) << "Created texture [" << texture_ << "]";

    // Bind to temporary unit.
    // workspace.BindTextureUnit(workspace.NumTextureUnits() - 1, texture_);
    glBindTexture(GL_TEXTURE_2D, texture_);

    // Similar to cudaMemcpy.
    OPENGL_CALL(glTexImage2D(GL_TEXTURE_2D, /*level=*/0, internal_format,
                width_, height_, /*border=*/0,
                format, GL_FLOAT, data));
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

void Download(GLfloat *data, GLint width, GLint height, GLuint texture){
    GLint ext_format, ext_type;
    glGetIntegerv(GL_IMPLEMENTATION_COLOR_READ_FORMAT, &ext_format);
    glGetIntegerv(GL_IMPLEMENTATION_COLOR_READ_TYPE, &ext_type);
    // OPENGL_CALL(glActiveTexture(GL_TEXTURE0 + tex_id));
    // OPENGL_CALL(glBindTexture(GL_TEXTURE_2D, texture));
    OPENGL_CALL(glReadBuffer(GL_COLOR_ATTACHMENT0));
    OPENGL_CALL(glReadPixels(0, 0, width, height, ext_format, ext_type, data));
}

void Download_DMA(GLfloat* data, GLint width, GLint height, GLuint texture){
    GLint ext_format, ext_type;
    glGetIntegerv(GL_IMPLEMENTATION_COLOR_READ_FORMAT, &ext_format);
    glGetIntegerv(GL_IMPLEMENTATION_COLOR_READ_TYPE, &ext_type);
    size_t bytes = width*height*sizeof(float);

    OPENGL_CALL(glReadBuffer(GL_COLOR_ATTACHMENT0));
    OPENGL_CALL(glBindBuffer(GL_PIXEL_PACK_BUFFER, texture));
    OPENGL_CALL(glBufferData(GL_PIXEL_PACK_BUFFER, bytes,
                NULL, GL_STREAM_READ));
    OPENGL_CALL(glReadPixels(0, 0, width, height, ext_format, ext_type,
                0));
    void* mem = glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, bytes, GL_READ_ONLY);
    assert(mem);
    memcpy(data, mem, bytes);
    glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
}

void SetInt(const std::string name, int value){
    glUniform1i(glGetUniformLocation(program, name.c_str()), value);
}

void ReadShader(const std::string fname){
    std::ifstream fd(fname);
    std::string src = std::string(std::istreambuf_iterator<char>(fd),
            (std::istreambuf_iterator<char>()));
    if(src.empty()){
        std::cerr<<"Read File ERROR from "<<fname;
    }
}

int main(int argc, char* argv[]){
    // Initialize Google's logging library.
    google::InitGoogleLogging(argv[0]);

    ::opengl::example::InitContext();

    CreateVertexShader();

    const int N = 100;
    GLint width = N;
    GLint height = N;
    const int niters = 100;
    const size_t num_elements = width * height * num_channels;
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> dist(1.0f, 2.0f);


    CreateProgram("../opengl/examples/fbo/fragment_es.glsl");

    //////////////////////////////////////////////////////
    // prepare data
    //////////////////////////////////////////////////////
    std::vector<GLfloat> texture0_data(num_elements, 0.25f);
    std::vector<GLfloat> texture1_data(num_elements, 0.25f);
    for (size_t i = 0; i != num_elements; ++i) {
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
    OPENGL_CALL(glViewport(0, 0, width, height));
    // Set "renderedTexture" as our colour attachement #0
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,
            output , 0);

    // Set the list of draw buffers.
    GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
    // "1" is the size of DrawBuffers.
    OPENGL_CALL(glDrawBuffers(1, DrawBuffers));

    // Always check that our framebuffer is ok
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        LOG(FATAL) << "Framebuffer not complete.";
    }

    // Tell the fragment shader what input textures to use.
    SetInput("A", input0, 0);
    SetInput("B", input1, 1);

    // set uniform
    SetInt("N", N);

    // swarmming up
    for (int iter = 0; iter < 10; ++iter) {
        OPENGL_CALL(glClear(GL_COLOR_BUFFER_BIT));
        OPENGL_CALL(glDrawArrays(GL_TRIANGLES, 0, 6));
        glFinish();
    }
    auto opengl_start = std::chrono::system_clock::now();
    for (int iter = 0; iter < niters; ++iter) {
        OPENGL_CALL(glClear(GL_COLOR_BUFFER_BIT));
        OPENGL_CALL(glDrawArrays(GL_TRIANGLES, 0, 6));
        glFinish();
    }


    auto opengl_end = std::chrono::system_clock::now();
    LOG(INFO) << "opengl: "
        << ((opengl_end - opengl_start).count() / niters);

    ///////////////////////////////////////
    // Download output Data from Device
    std::vector<GLfloat> retrieved_data(static_cast<size_t>(num_elements));
    Download(retrieved_data.data(), width, height, output);

    std::vector<GLfloat> cpu_result(static_cast<size_t>(num_elements));
    auto cpu_start = std::chrono::system_clock::now();
    for (int iter = 0; iter < niters; ++iter) {
        for (int row = 0; row != N; ++row) {
            for (int col = 0; col != N; ++col) {
                cpu_result[(row * N + col)*num_channels] = 0.0f;
                for (int i = 0; i != N; ++i) {
                    GLfloat a = texture0_data[(row * N + i)*num_channels];
                    GLfloat b = texture1_data[(i * N + col)*num_channels];
                    cpu_result[(row * N + col)*num_channels] += a * b;
                }
            }
        }
    }
    auto cpu_end = std::chrono::system_clock::now();

    // just test instead of print for big matrix Multiplication
    for (size_t i = 0; i < retrieved_data.size(); ++i) {
        if(std::abs(retrieved_data[i] - cpu_result[i]) > 0.001f){
            LOG(FATAL)<<"Expect value: "<<cpu_result[i]
                <<" Actural value: "<<retrieved_data[i]
                <<" in Index: "<<i<<std::endl;
        }
    }

    LOG(INFO) << "cpu:    "
        << ((cpu_end - cpu_start).count() / niters);

    glDeleteFramebuffers(1, &frame_buffer);

    ::opengl::example::DestroyContext();
    return 0;
}

