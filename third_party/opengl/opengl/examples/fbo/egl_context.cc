#include "opengl/examples/fbo/context.h"

#include <EGL/egl.h>

namespace opengl{
    namespace {
        // egl
        EGLContext egl_context;
        EGLDisplay egl_display;
        EGLSurface egl_surface;
    }//namespace

    namespace example{
        void InitContext(){
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
            // int major;

            LOG(INFO) << "Opengl says version: " << glGetString(GL_VERSION);

            OPENGL_CHECK_ERROR;
        }

        void DestroyContext(){
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
    }//namespace example
}//namespace opengl




