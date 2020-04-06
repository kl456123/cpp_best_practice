#include <stdio.h>
#include <memory>

#include "program.h"
#include "glut.h"
#include "buffer.h"
#include "context.h"


int main(int argc, char** argv){
    /////////////////////////////////////
    // init and get statisic ////////////
    /////////////////////////////////////
    InitGLUT(argc, argv);
    int maxtexsize;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE,&maxtexsize);
    printf("GL_MAX_TEXTURE_SIZE, %d\n",maxtexsize);

    // prepare program
    const char source[] = "";

    Program program;
    program.Attach(source)
        .Link();

    program.Activate();

    // prepare runtime
    Context context(nullptr);

    //prepare input and output
    ShaderBuffer input(1<<5);
    ShaderBuffer output(1<<5);

    context.Compute({1,2,3});
    context.Finish();

    return 0;
}
