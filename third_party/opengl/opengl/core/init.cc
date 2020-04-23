#include <iostream>
#include <glog/logging.h>
#include "init.h"

namespace opengl{

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




#ifdef ARM_PLATFORM
    void egl_init(){
        // init for embedding platform
        // just assign for the following variable
        EGLContext egl_context;
        EGLDisplay egl_display;
        EGLSurface egl_surface;

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
        glGetIntegerv(GL_MAJOR_VERSION, &major);
        LOG(INFO)<<"current opengl version: "<<major;
    }
#else
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
            LOG(FATAL)<<"Failed to Create OpenGL Context";
        }

        // Create Context and Load OpenGL Functions
        glfwMakeContextCurrent(window);
        return window;
    }

#endif


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
}//namespace opengl
