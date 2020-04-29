/**************************************
 * This examples is to demostrate the use of egl to run in embedding system
 * It's mainly different from glfw during initialization time.
 */
#include <iostream>
#include <string>
#include <sstream>
#include <glog/logging.h>

// opengles headers
#include <EGL/egl.h>
#ifdef USE_GLEW
#include <GL/glew.h>
#else
#include <GLES2/gl2.h>
#endif

EGLContext egl_context;
EGLDisplay egl_display;
EGLSurface egl_surface;

void EGLInit(){
    // init for embedding platform
    // just assign for the following variable

    if(eglGetCurrentContext()==EGL_NO_CONTEXT){
        egl_context = EGL_NO_CONTEXT;
        VLOG(1)<<"No Current Context Found! Need to Create Again";
    }

    egl_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (egl_display == EGL_NO_DISPLAY) {
        LOG(FATAL)<<"eglGetDisplay Failed When Creating Context!";
    }
    int majorVersion;
    int minorVersion;
    eglInitialize(egl_display, &majorVersion, &minorVersion);
    EGLint numConfigs;
    static const EGLint configAttribs[] = {EGL_SURFACE_TYPE,
        EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE,
        EGL_OPENGL_ES2_BIT,
        EGL_RED_SIZE,
        8,
        EGL_GREEN_SIZE,
        8,
        EGL_BLUE_SIZE,
        8,
        EGL_ALPHA_SIZE,
        8,
        EGL_NONE};

    EGLConfig surfaceConfig;
    if(!eglChooseConfig(egl_display, configAttribs, &surfaceConfig, 1, &numConfigs)){
        eglMakeCurrent(egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        eglTerminate(egl_display);
        egl_display = EGL_NO_DISPLAY;
        LOG(FATAL)<<"eglChooseConfig Failed When Creating Context!";
    }

    static const EGLint contextAttribs[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
    egl_context                             = eglCreateContext(egl_display, surfaceConfig, NULL, contextAttribs);
    static const EGLint surfaceAttribs[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
    egl_surface                             = eglCreatePbufferSurface(egl_display, surfaceConfig, surfaceAttribs);
    eglMakeCurrent(egl_display, egl_surface, egl_surface, egl_context);
    eglBindAPI(EGL_OPENGL_ES_API);
    int major;
    // glGetIntegerv(GL_MAJOR_VERSION, &major);
    // LOG(INFO)<<"current opengl version: "<<major;
}

void outputGLESInfo() {
    std::cout << "GL_VENDOR = " << glGetString(GL_VENDOR) << "\n";
    std::cout << "GL_RENDERER = " << glGetString(GL_RENDERER) << "\n";
    std::cout << "GL_VERSION = " << glGetString(GL_VERSION) << "\n";
    std::cout << "GL_SHADING_LANGUAGE_VERSION = " << glGetString(GL_SHADING_LANGUAGE_VERSION) << "\n";
    std::cout << "Extensions :\n";
    std::string extBuffer;
    std::stringstream extStream;
    extStream << glGetString(GL_EXTENSIONS);
    while (extStream >> extBuffer) {
        std::cout << extBuffer << "\n";
    }

}

void EGLDestroy(){
    if (egl_display != EGL_NO_DISPLAY) {
        if (egl_context != EGL_NO_CONTEXT) {
            eglDestroyContext(egl_display, egl_context);
            egl_context = EGL_NO_CONTEXT;
        }
        if (egl_surface != EGL_NO_SURFACE) {
            eglDestroySurface(egl_display, egl_surface);
            egl_surface = EGL_NO_SURFACE;
        }
        eglMakeCurrent(egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        eglTerminate(egl_display);
        egl_display = EGL_NO_DISPLAY;
    }
    eglReleaseThread();
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

const float* MockData(int* size){
    // mem in cpu
    // triangle
    static float vertices[] = {
        -0.5f, -0.5f, 0.0f,
        0.5f, -0.5f, 0.0f,
        0.0f, 0.5f, 0.0f
    };

    return vertices;
}

int main(int arc, char* argv[]){
    google::InitGoogleLogging(argv[0]);
    EGLInit();
    ////////////////////////////////////
    // program run here

    int size=0;
    const float* vertices = MockData(&size);
    InitVBO(vertices, size);

    //////////////////////////////////////
    EGLDestroy();
    return 0;
}

