#include "kernels/binary.h"

#include "program.h"
#include "context.h"


BinaryKernel::BinaryKernel(Context* context)
    :context_(context){
        // set work size
        for(int i=0;i<3;i++){
            work_sizes_[i] = 1;
        }

        // set program
        program_ = new Program;
        std::string fname = "../examples/glsl/binary.glsl";
        program_->Attach(fname);
        program_->Link();
    }


BinaryKernel::~BinaryKernel(){
    if(program_!=nullptr){delete program_;}
}

void BinaryKernel::Compute(TensorList& inputs, TensorList& outputs){
    // use program first
    program_->Activate();

    // set input and output
    auto input0 = inputs[0];
    auto input1 = inputs[1];
    auto output = outputs[0];
    auto shape = input0->dims();
    int ih = shape[1];
    int iw = shape[2];
    int ic = shape[3];
    int ic_4 = ic/4;

    glBindImageTexture(0, output->template device_id<Texture>(), 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA32F);
    {
        int text_id = 0;
        glActiveTexture(GL_TEXTURE0 + text_id);
        glUniform1i(1, text_id);
        glBindTexture(GL_TEXTURE_3D, input0->template device_id<Texture>());
    }

    {
        int text_id = 1;
        glActiveTexture(GL_TEXTURE0 + text_id);
        glUniform1i(2, text_id);
        glBindTexture(GL_TEXTURE_3D, input1->template device_id<Texture>());
    }

    glUniform4i(3, iw, ih, ic_4, 1);

    // run
    context_->Compute({iw/work_sizes_[0], ih/work_sizes_[1], ic_4/work_sizes_[2]});
}



