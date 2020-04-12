#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#include <GL/glew.h>
#include <GL/glut.h>

#define ERROR_GLEW -1

// global variables
GLuint fb;

void initGLUT(int argc, char** argv){
    glutInit(&argc, argv);
    glutCreateWindow("SAXPY TESTS");
}

void initGLEW(void){
    int err = glewInit();

    if(GLEW_OK !=err){
        // printf((char*)glewGetErrorString(err));
        exit(ERROR_GLEW);
    }
}

void initFBO(void){
    // create FBO (off-screen framebuffer)
    glGenFramebuffersEXT(1, &fb);

    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fb);
}


int main(int argc, char**argv){
    initGLUT(argc, argv);
    initGLEW();

    int maxtexsize;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxtexsize);
    printf("GL_MAX_TEXTURE_SIZE, %d\n", maxtexsize);

    int N = 100;
    float* dataY = (float*)malloc(N*sizeof(float));
    float* dataX = (float*)malloc(N*sizeof(float));
    float alpha;
    int texSize = sqrt(N);

    // create texture
    GLuint texID;
    GLenum texture_target = GL_TEXTURE_2D;
    GLenum internal_format = GL_RGBA32F;
    GLenum texture_format = GL_RGBA;
    glGenTextures(1, &texID);
    glBindTexture(texture_target, texID);
    glTexParameteri(texture_target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(texture_target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(texture_target, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(texture_target, GL_TEXTURE_WRAP_T, GL_CLAMP);

    // WHY
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glTexImage2D(texture_target, 0, internal_format, texSize,
            texSize, 0, texture_format, GL_FLOAT, 0);

    return 0;
}
