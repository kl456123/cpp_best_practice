/*****************************************
 * This file is to demostrate the usage of custom framebuffer.
 * Main work of total computation is done in fragment shader
 * so that we can do gpgpu in it.
 */
#include <iostream>

#include <GL/glew.h>
#include <GLFW/glfw3.h>


int main(){
    unsigned int fbo;
    glGenFramebuffers(1, &fbo);

    GLenum frame_target = GL_FRAMEBUFFER;
    glBindFramebuffer(frame_target, fbo);

    // create texture
    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 800, 600, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // attach texture to framebuffer
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);

    // check framebuffer
    if(glCheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE){
        // error handler
        std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" << std::endl;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
    return 0;
}
