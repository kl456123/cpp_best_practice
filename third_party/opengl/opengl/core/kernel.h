#ifndef OPENGL_CORE_KERNEL_H_
#define OPENGL_CORE_KERNEL_H_
#include <string>

#include "opengl/core/opengl.h"
#include "opengl/core/types.h"
#include "opengl/core/dlxnet.pb.h"

namespace opengl{
    class Program;
    class Tensor;
    class Context;

    class Kernel{
        public:
            Kernel(Context* context);
            virtual ~Kernel();

            virtual void SetupProgram(GLuint vertex_shader);
            /*!
             * Run Kernel, do computation actually
             */
            virtual void Compute(TensorList& inputs, TensorList& outputs)=0;

            virtual void SetupAttr(const dlxnet::Attribute& attr)=0;

            void Compute(){
                Compute(input_tensors_, output_tensors_);
            }

            DataFormat GetOutputDFormat(int i)const;

            /*!
             * Compute output shapes according to their input tensor shape
             */
            virtual void InferOutputShape(TensorShapeList& inputs,
                    TensorShapeList& outputs)=0;

            // some accessors
            void set_kernel_name(std::string name){
                kernel_name_ = name;
            }
            void set_kernel_type(std::string name){
                kernel_type_ = name;
            }
            std::string kernel_name()const{
                return kernel_name_;
            }
            std::string kernel_type()const{
                return kernel_type_;
            }
        protected:
            // attach output tensor to the target(fbo)
            // used in compute function of subclass
            void SetFrameBuffer(TensorList& outputs);

            void SetVertexShader();

            // kernel program(opencl) or shader(opengl)
            Program* program_;

            // opengl driver, it wrapping all API about platform(opengl or opencl)
            Context* context_;

            // filename of kernel source file
            std::string kernel_fname_;

            // global works size and local work size
            unsigned long work_sizes_[3];

            // store input and output indexes
            std::vector<Tensor*> input_tensors_;
            std::vector<Tensor*> output_tensors_;

            std::vector<int> input_tensor_indexes_;
            std::vector<int> output_tensor_indexes_;

            std::vector<DataFormat> output_tensor_dformats_;

            // make it can fill input and output tensors
            friend class FBOSession;

            std::string kernel_name_;
            std::string kernel_type_;
    };
}//namespace opengl


#endif
