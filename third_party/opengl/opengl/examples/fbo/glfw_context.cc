#include "opengl/examples/fbo/context.h"
#define GLFW_INCLUDE_ES31
#include <GLFW/glfw3.h>

namespace opengl{
    namespace {
        GLFWwindow* window_;
        // global variables here
        const int kWindowWidth = 640;
        const int kWindowHeight = 480;

        void GlfwErrorCallback(int err, const char *str) {
            LOG(ERROR) << "Error: [" << err << "] " << str;
        }
    }//namespace

    namespace example{
        void InitContext(){
            // Set an error handler.
            // This can be called before glfwInit().
            glfwSetErrorCallback(&GlfwErrorCallback);

            // Initialize GLFW.
            if (glfwInit() != GL_TRUE) {
                LOG(FATAL) << "glfwInit() failed!" << std::endl;
            }

            // Create a window.
            // TODO(zhixunt): GLFW allows us to create an invisible window.
            // TODO(zhixunt): On retina display, window size is different from framebuffer size.
            glfwWindowHint(GLFW_CLIENT_API,  GLFW_OPENGL_ES_API);
            glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
            glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
            glfwWindowHint(GLFW_VISIBLE, GL_FALSE);
            window_ = glfwCreateWindow(kWindowWidth, kWindowHeight, "", nullptr, nullptr);
            if (window_ == nullptr) {
                LOG(FATAL) << "glfwCreateWindow() failed!";
            }

            LOG(INFO) << "GLFW says OpenGL version: "
                << glfwGetWindowAttrib(window_, GLFW_CONTEXT_VERSION_MAJOR)
                << "."
                << glfwGetWindowAttrib(window_, GLFW_CONTEXT_VERSION_MINOR)
                << "."
                << glfwGetWindowAttrib(window_, GLFW_CONTEXT_REVISION);

            // Before using any OpenGL API, we must specify a context.
            glfwMakeContextCurrent(window_);

            LOG(INFO) << "Opengl says version: " << glGetString(GL_VERSION);
            OPENGL_CHECK_ERROR;
        }

        void DestroyContext(){
            // Paired with glfwCreateWindow().
            glfwDestroyWindow(window_);

            // Paired with glfwInit().
            glfwTerminate();
        }
    }//namespace example
}//namespace opengl
