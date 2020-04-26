/*********************************************************
 * better implementation for NVIDIA
 * Use FBO to download and upload data between cpu and gpu
 */
#include <stdio.h>
#include <stdlib.h>
#include <GL/glew.h>
#include <GL/glut.h>

// global vars
GLhandleARB programObject;
GLhandleARB shaderObject;
GLint yParam, xParam, alphaParam;


void initCG(){
}

void initGLSL(const char* program_source){
    // create program object
    // create program object
    GLuint programObject = glCreateProgramObjectARB();
    // create shader object (fragment shader) and attach to program
    GLuint shaderObject = glCreateShaderObjectARB(GL_FRAGMENT_SHADER_ARB);
    glAttachObjectARB (programObject, shaderObject);
    // set source to shader object
    glShaderSourceARB(shaderObject, 1, &program_source, NULL);
    // compile
    glCompileShaderARB(shaderObject);
    // link program object together
    glLinkProgramARB(programObject);
    // Get location of the texture samplers for future use
    yParam = glGetUniformLocationARB(programObject, "textureY");
    xParam = glGetUniformLocationARB(programObject, "textureX");
    alphaParam = glGetUniformLocationARB(programObject, "alpha");
}

int main(int argc, char **argv) {
    // declare texture size, the actual data will be a vector
    // of size texSize*texSize*4
    int texSize = 2;
    // create test data
    float* data = (float*)malloc(4*texSize*texSize*sizeof(float));
    float* result = (float*)malloc(4*texSize*texSize*sizeof(float));
    for (int i=0; i<texSize*texSize*4; i++)
        data[i] = i+1.0;
    // set up glut to get valid GL context and
    // get extension entry points
    glutInit (&argc, argv);
    glutCreateWindow("TEST1");
    glewInit();
    // viewport transform for 1:1 pixel=texel=data mapping
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0,texSize,0.0,texSize);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glViewport(0,0,texSize,texSize);
    // create FBO and bind it (that is, use offscreen render target)
    GLuint fb;
    glGenFramebuffersEXT(1,&fb);
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT,fb);
    // create texture
    GLuint tex;
    glGenTextures (1, &tex);
    glBindTexture(GL_TEXTURE_RECTANGLE_ARB,tex);
    // set texture parameters
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB,
            GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB,
            GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB,
            GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_RECTANGLE_ARB,
            GL_TEXTURE_WRAP_T, GL_CLAMP);
    // define texture with floating point format
    glTexImage2D(GL_TEXTURE_RECTANGLE_ARB,0,GL_RGBA32F_ARB,
            texSize,texSize,0,GL_RGBA,GL_FLOAT,0);
    // attach texture
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT,
            GL_COLOR_ATTACHMENT0_EXT,
            GL_TEXTURE_RECTANGLE_ARB,tex,0);
    // transfer data to texture
    glTexSubImage2D(GL_TEXTURE_RECTANGLE_ARB,0,0,0,texSize,texSize,
            GL_RGBA,GL_FLOAT,data);
    /////////////////////////////////////////
    // Computing = Drawing
    /////////////////////////////////////////

    // prepare prarameters
    // enable texture y_old (read-only)
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(textureParameters.texTarget,yTexID[readTex]);
    glUniform1iARB(yParam,0); // texunit 0
    // enable texture x (read-only)
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(textureParameters.texTarget,xTexID);
    glUniform1iARB(xParam, 1); // texunit 1
    // enable scalar alpha
    glUniform1fARB(alphaParam,alpha);

    // performing the computation


    // and read back
    glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);
    glReadPixels(0, 0, texSize, texSize,GL_RGBA,GL_FLOAT,result);
    // print out results
    printf("Data before roundtrip:\n");
    for (int i=0; i<texSize*texSize*4; i++)
        printf("%f\n",data[i]);
    printf("Data after roundtrip:\n");
    for (int i=0; i<texSize*texSize*4; i++)
        printf("%f\n",result[i]);
    // clean up
    free(data);
    free(result);
    glDeleteFramebuffersEXT (1,&fb);
    glDeleteTextures (1,&tex);
    return 0;
}
