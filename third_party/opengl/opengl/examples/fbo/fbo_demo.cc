/* This file demostrate how to use framebuffer object
 * to do gpgpu computation task, especial in some case
 * that we cannot use compute shader in opengles-3.0 or
 * more old version.
 * But in some case the demo file will also fail due to
 * lack of some features, like `glFramebufferTexture`, the
 * api cannot be used in some device or runtime. so be careful
 * when error happens in `OPENGL_CHECK_ERROR`
 *
 *
 *
 *
 */

#include <string.h>
#include <fstream>

#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>
#include <vector>
#include <random>
#include <glog/logging.h>

#include "opengl/examples/fbo/context.h"
#define BUFFER_OFFSET(i) ((char *)NULL + (i))

GLuint vertex_shader_;
GLuint fragment_shader;
GLuint program;
GLuint frame_buffer;

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

void SetVertexShader(){
    auto point_attrib = GLuint(glGetAttribLocation(program, "point"));
    OPENGL_CALL(glEnableVertexAttribArray(point_attrib));

    OPENGL_CALL(glVertexAttribPointer(point_attrib, 2, GL_FLOAT, GL_FALSE,
                sizeof(Vertex), nullptr));
}

void SetFrameBuffer(int height, int width, GLuint output_texture){
    OPENGL_CALL(glViewport(0, 0, width, height));

    // Set "renderedTexture" as our colour attachement #0
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
            output_texture , 0);

    // Always check that our framebuffer is ok
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        LOG(FATAL) << "Framebuffer not complete.";
    }
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
}

GLuint CreateTexture(const GLfloat *data, GLsizei width,
        GLsizei height, GLsizei depth){
    GLuint texture_;
    // Create a texture.
    OPENGL_CALL(glGenTextures(1, &texture_));
    LOG(INFO) << "Created texture [" << texture_ << "]";
    glBindTexture(GL_TEXTURE_3D, texture_);

    OPENGL_CALL(glTexImage3D(GL_TEXTURE_3D, /*level=*/0, internal_format,
                width, height, depth, /*border=*/0,
                format, GL_FLOAT, nullptr));

    OPENGL_CALL(
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    OPENGL_CALL(
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
    OPENGL_CALL(
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE));
    OPENGL_CALL(
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
    OPENGL_CALL(
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));

    if(data){
        OPENGL_CALL(glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0,0, width, height, depth,
                    format, GL_FLOAT, data));
    }
    return texture_;
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
                format, GL_FLOAT, nullptr));
    // TODO(zhixunt): What are these?
    OPENGL_CALL(
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    OPENGL_CALL(
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
    OPENGL_CALL(
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
    OPENGL_CALL(
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));

    if(data){
        OPENGL_CALL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_, height_,
                    format, GL_FLOAT, data));
    }

    return texture_;
}

void SetInput2D( std::string name, GLuint id,  int tex_id){
    GLint location= glGetUniformLocation(program, name.c_str());
    glUniform1i(location, tex_id);
    glActiveTexture(GL_TEXTURE0+tex_id);
    glBindTexture(GL_TEXTURE_2D, id);
}

void SetInput3D( std::string name, GLuint id,  int tex_id){
    GLint location= glGetUniformLocation(program, name.c_str());
    glUniform1i(location, tex_id);
    glActiveTexture(GL_TEXTURE0+tex_id);
    glBindTexture(GL_TEXTURE_3D, id);
}

void CheckFormatAndType(GLint ext_format, GLint ext_type){
}

void Download(GLfloat *data, GLint width, GLint height, GLuint texture){
    GLint ext_format, ext_type;
    glGetIntegerv(GL_IMPLEMENTATION_COLOR_READ_FORMAT, &ext_format);
    glGetIntegerv(GL_IMPLEMENTATION_COLOR_READ_TYPE, &ext_type);
    CHECK_EQ(ext_type, GL_FLOAT)<<"unmatched type";
    CHECK_EQ(ext_format, format)<<"unmatched format";
    // OPENGL_CALL(glActiveTexture(GL_TEXTURE0 + tex_id));
    // OPENGL_CALL(glBindTexture(GL_TEXTURE_2D, texture));
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
            texture , 0);
    OPENGL_CALL(glReadBuffer(GL_COLOR_ATTACHMENT0));
    OPENGL_CALL(glReadPixels(0, 0, width, height, ext_format, ext_type, data));
}

void Upload(GLfloat* data, GLuint texture, GLint width, GLint height){
    OPENGL_CALL(glBindTexture(GL_TEXTURE_2D, texture));
    OPENGL_CALL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                width, height, format, GL_FLOAT, data));
}

void Upload_DMA(GLfloat* data, GLuint texture, GLint width, GLint height){
    GLuint io_buffer;
    OPENGL_CALL(glGenBuffers(1, &io_buffer));
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, io_buffer);
    size_t bytes = width*height*num_channels*sizeof(float);
    OPENGL_CALL(glBufferData(GL_PIXEL_UNPACK_BUFFER, bytes, NULL, GL_STREAM_DRAW));

    // copy data from host to pbo
    void* mem = glMapBufferRange(GL_PIXEL_UNPACK_BUFFER, 0, bytes, GL_MAP_WRITE_BIT);
    CHECK_NOTNULL(mem);
    memcpy(mem, data, bytes);
    OPENGL_CALL(glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER));
    // copy pbo to texture
    OPENGL_CALL(glBindTexture(GL_TEXTURE_2D, texture));
    OPENGL_CALL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                width, height, format, GL_FLOAT, BUFFER_OFFSET(0)));
    OPENGL_CALL(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0));
}

void Download_DMA(GLfloat* data, GLint width, GLint height, GLuint texture){
    GLint ext_format, ext_type;
    glGetIntegerv(GL_IMPLEMENTATION_COLOR_READ_FORMAT, &ext_format);
    glGetIntegerv(GL_IMPLEMENTATION_COLOR_READ_TYPE, &ext_type);
    size_t bytes = width*height*num_channels*sizeof(float);
    GLuint io_buffer;
    OPENGL_CALL(glGenBuffers(1, &io_buffer));

    // specify which texture to read
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
            texture , 0);

    OPENGL_CALL(glReadBuffer(GL_COLOR_ATTACHMENT0));
    OPENGL_CALL(glBindBuffer(GL_PIXEL_PACK_BUFFER, io_buffer));
    OPENGL_CALL(glBufferData(GL_PIXEL_PACK_BUFFER, bytes,
                NULL, GL_STREAM_READ));
    // copy data from fbo to pbo
    OPENGL_CALL(glReadPixels(0, 0, width, height, ext_format, ext_type,
                BUFFER_OFFSET(0)));
    // copy data from pbo to host
    void* mem = glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, bytes, GL_MAP_READ_BIT);
    CHECK_NOTNULL(mem);
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

void InitFrameBuffer(){
    OPENGL_CALL(glGenFramebuffers(1, &frame_buffer));

    // Create frame buffer And Check its Completation
    OPENGL_CALL(glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer));
    // Set the list of draw buffers.
    GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
    // "1" is the size of DrawBuffers.
    OPENGL_CALL(glDrawBuffers(1, DrawBuffers));
}

void DestoryFrameBuffer(){
    glDeleteFramebuffers(1, &frame_buffer);
}

void Run(std::vector<int> shape){
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> dist(1.0f, 2.0f);

    const int width=shape[1];
    const int height = shape[0];
    const int num_channels = shape[2];
    const int N = height;
    CHECK_EQ(height, width)<<"Only square input is supported now";

    const size_t num_elements = width * height * num_channels;

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
    //
    auto input0 = CreateTexture(nullptr, width, height);
    auto input1 = CreateTexture(nullptr, width, height);

    Upload_DMA(texture0_data.data(), input0, width, height);
    Upload_DMA(texture1_data.data(), input1, width, height);

    auto output = CreateTexture(nullptr, width, height);

    // auto output2 = CreateTexture(nullptr, width, height, 10);

    ////////////////////////////////////////
    // Compute(Render)
    ////////////////////////////////////////
    OPENGL_CALL(glUseProgram(program));


    // Tell the fragment shader what input textures to use.
    SetFrameBuffer(height, width, output);
    SetVertexShader();
    SetInput2D("A", input0, 0);
    SetInput2D("B", input1, 1);

    // set uniform
    SetInt("N", N);

    // auto opengl_start = std::chrono::system_clock::now();
    OPENGL_CALL(glClear(GL_COLOR_BUFFER_BIT));
    OPENGL_CALL(glDrawArrays(GL_TRIANGLES, 0, 6));
    glFinish();


    // auto opengl_end = std::chrono::system_clock::now();
    // LOG(INFO) << "opengl: "
    // << ((opengl_end - opengl_start).count() / niters);

    ///////////////////////////////////////
    // Download output Data from Device
    std::vector<GLfloat> retrieved_data(static_cast<size_t>(num_elements));
    Download_DMA(retrieved_data.data(), width, height, output);
    // std::vector<GLfloat> retrieved_data2(static_cast<size_t>(num_elements));
    // Download(retrieved_data2.data(), width, height, input0);

    std::vector<GLfloat> cpu_result(static_cast<size_t>(num_elements));
    // auto cpu_start = std::chrono::system_clock::now();
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
    // auto cpu_end = std::chrono::system_clock::now();

    // just test instead of print for big matrix Multiplication
    for (size_t i = 0; i < retrieved_data.size(); ++i) {
        if(std::abs(retrieved_data[i] - cpu_result[i]) > 0.001f){
            LOG(FATAL)<<"Expect value: "<<cpu_result[i]
                <<" Actural value: "<<retrieved_data[i]
                <<" in Index: "<<i<<std::endl;
        }
    }

    // LOG(INFO) << "cpu:    "
    // << ((cpu_end - cpu_start).count() / niters);
}

void LogSystemInfo(){
    printf("OpenGL info:\n"
            "\tVendor   = \"%s\"\n"
            "\tRenderer = \"%s\"\n"
            "\tVersion  = \"%s\"\n"
            "\tGLSL     = \"%s\"\n",
            glGetString(GL_VENDOR),
            glGetString(GL_RENDERER),
            glGetString(GL_VERSION),
            glGetString(GL_SHADING_LANGUAGE_VERSION)
          );
}

int main(int argc, char* argv[]){
    // Initialize Google's logging library.
    google::InitGoogleLogging(argv[0]);

    ::opengl::example::InitContext();
    LogSystemInfo();
    InitFrameBuffer();

    CreateVertexShader();

    // some params
    const int N = 100;
    GLint width = N;
    GLint height = N;
    const int niters = 100;

    for(int i=0;i<niters;++i){
        Run({height, width, num_channels});
    }


    DestoryFrameBuffer();
    ::opengl::example::DestroyContext();

    LOG(INFO)<<"No Error Found!";
    return 0;
}

