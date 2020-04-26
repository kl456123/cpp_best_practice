/*****************************************
 * This file is to demostrate the usage of custom framebuffer.
 * Main work of total computation is done in fragment shader
 * so that we can do gpgpu in it.
 */
#include <GL/glew.h>
#include <GLFW/glfw3.h>


int main(){
    unsigned int fbo;
    glGenFramebuffers(1, &fbo);

    GLenum frame_target = GL_FRAMEBUFFER;
    glBindFramebuffer(frame_target, fbo);
    return 0;
}
