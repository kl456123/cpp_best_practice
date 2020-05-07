#include "opengl/core/fbo_session.h"
#include "opengl/utils/macros.h"
#include "opengl/core/kernel.h"
#include "opengl/core/kernel_registry.h"
#include "opengl/utils/protobuf.h"
#include "opengl/core/tensor.h"


namespace opengl{
    namespace{
        const GLenum kDataType=GL_FLOAT;
        GLenum kInternalFormat = GL_RGBA32F;
        GLenum kFormat = GL_RGBA;
        // Don't need to change this.
        // We want to draw 2 giant triangles that cover the whole screen.
        struct Vertex {
            float x, y;
        };

        static constexpr size_t kNumVertices = 6;

        const char *vertex_shader_text = "#version 300 es\n"
            "in vec2 point; // input to vertex shader\n"
            "void main() {\n"
            "  gl_Position = vec4(point, 0.0, 1.0);\n"
            "}\n";

        const Vertex vertices[kNumVertices] = {
            {-1.f, -1.f},
            {1.0f, -1.f},
            {1.0f, 1.0f},
            {-1.f, -1.f},
            {-1.f, 1.0f},
            {1.0f, 1.0f},
        };
    }//namespace

    FBOSession::~FBOSession(){}

    void FBOSession::LoadGraph(const std::string file_path){
        // load graph from disk
        CHECK(ReadProtoFromBinary(file_path.c_str(), model_))
            <<"Load Graph "<<file_path <<"Failed";

        dlxnet::GraphProto graph = model_->graph();
        // allocate memory for each tensor
        // create kernel for each node

        for(auto& tensor_name:graph.tensor_names()){
            total_tensor_names_.emplace_back(tensor_name);
        }

        Kernel* kernel=nullptr;
        for(auto& node: graph.node()){
            KernelRegistry::Global()->CreateKernel(node.type(), &kernel, context_);
            if(kernel==nullptr){
                LOG(FATAL)<<"unsupported kernel name "<<node.type();
            }
            // setup program for each kernel here
            kernel->SetupProgram(vertex_shader_);

            kernel->SetupAttr(node.attr());
            // fill inputs and outputs
            for(int i=0;i<node.input_index_size();++i){
                Tensor* input_tensor = total_tensors_[node.input_index(i)];
                kernel->input_tensors_.emplace_back(input_tensor);
            }
            for(int i=0;i<node.output_index_size();++i){
                Tensor* output_tensor = total_tensors_[node.output_index(i)];
                kernel->output_tensors_.emplace_back(output_tensor);
            }
            kernels_.emplace_back(kernel);
        }
    }

    void FBOSession::CreateVertexShader(){
        // We always render the same vertices and triangles.
        GLuint vertex_buffer;
        OPENGL_CALL(glGenBuffers(1, &vertex_buffer));
        OPENGL_CALL(glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer));
        OPENGL_CALL(glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices,
                    GL_STATIC_DRAW));

        GLuint vertex_array;
        OPENGL_CALL(glGenVertexArrays(1, &vertex_array));
        OPENGL_CALL(glBindVertexArray(vertex_array));
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);

        // We always use the same vertex shader.
        vertex_shader_ = CreateShader(GL_VERTEX_SHADER, vertex_shader_text);
    }

    GLuint FBOSession::CreateShader(GLenum shader_kind, const char *shader_src) {
        // Create the shader.
        GLuint shader = glCreateShader(shader_kind);
        glShaderSource(shader, 1, &shader_src, nullptr);
        glCompileShader(shader);

        // Check compile errors.
        GLint err;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &err);

        GLint info_log_len;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &info_log_len);

        if (info_log_len > 0) {
            std::unique_ptr<char[]> err_msg(new char[info_log_len + 1]);
            glGetShaderInfoLog(shader, info_log_len, nullptr, err_msg.get());
            LOG(FATAL) << err_msg.get();
        }

        OPENGL_CHECK_ERROR;

        return shader;
    }

    void FBOSession::Run(){
        for(int i=0;i<kernels_.size();++i){
            // prepare input and output
            // set output here
            // SetFrameBuffer(output_tensors_per_kernel);
            kernels_[i]->Compute();
        }
    }


    FBOSession::FBOSession(Context* context)
        :context_(context){
            // create vertex shader first
            CreateVertexShader();

            model_ = new dlxnet::ModelProto;
        }

    void FBOSession::AllocateTensor(const TensorShapeList& shapes, TensorList& tensors){
        // tensors.resize(shapes.size());
        tensors.clear();
        for(auto& shape: shapes){
            // only allocate texture tensor in session
            // due to that only texture 2d tensor is used to feed or output
            tensors.emplace_back(new Tensor(Tensor::DT_FLOAT, shape,
                        Tensor::DEVICE_TEXTURE));
        }
    }

    void FBOSession::Setup(TensorList inputs_cpu){
        CHECK(kernels_.size())<<"Graph is empty, please load it first";

        // allocate memory for each kernel here
        TensorShapeList input_shapes, output_shapes;

        // prepare input and output tensors first
        output_tensors_.resize(kernels_.size());
        input_tensors_.resize(kernels_.size());

        for(int i=0;i<kernels_.size();++i){
            auto& texture_inputs_per_kernel = input_tensors_[i];

            // calculate output shapes for each kernel
            for(auto& input_cpu:inputs_cpu){
                // here we just upload input_cpu to input_gpu, and
                // set it to the first kernel in the session
                CHECK(input_cpu->is_host());
                auto texture_input = new Tensor(Tensor::DT_FLOAT, input_cpu->shape(),
                        Tensor::DEVICE_TEXTURE);
                Upload(input_cpu, texture_input);
                input_shapes.emplace_back(input_cpu->shape());
                texture_inputs_per_kernel.emplace_back(texture_input);
            }
            kernels_[i]->InferOutputShape(input_shapes,
                    output_shapes);
            input_shapes = output_shapes;

            // then allocate them
            AllocateTensor(output_shapes, output_tensors_[i]);
        }
    }

    void FBOSession::LoadGraph(StringList kernel_names){
        // init kernels first
        kernels_.clear();
        kernels_.reserve(kernel_names.size());

        // create each kernel
        Kernel* kernel=nullptr;
        for(auto& kernel_name:kernel_names){
            KernelRegistry::Global()->CreateKernel(kernel_name, &kernel, context_);
            if(kernel==nullptr){
                LOG(FATAL)<<"unsupported kernel name "<<kernel_name;
            }
            // setup program for each kernel here
            kernel->SetupProgram(vertex_shader_);

            kernels_.emplace_back(kernel);
        }
    }

    void FBOSession::Download(Tensor* tensor){
        GLint ext_format, ext_type;
        const int width = tensor->shape()[0];
        const int height = tensor->shape()[1];
        glGetIntegerv(GL_IMPLEMENTATION_COLOR_READ_FORMAT, &ext_format);
        glGetIntegerv(GL_IMPLEMENTATION_COLOR_READ_TYPE, &ext_type);
        CHECK_EQ(ext_type, kDataType)<<"unmatched type";
        CHECK_EQ(ext_format, kFormat)<<"unmatched format";

        // download
        OPENGL_CALL(glReadBuffer(GL_COLOR_ATTACHMENT0));
        OPENGL_CALL(glReadPixels(0, 0, width, height, ext_format, ext_type, tensor->host()));
    }

    void FBOSession::Upload(Tensor* cpu_tensor, Tensor* device_tensor){
        const int width = cpu_tensor->shape()[0];
        const int height = cpu_tensor->shape()[1];
        OPENGL_CALL(glBindTexture(GL_TEXTURE_2D, device_tensor->device<Texture>()->id()));
        OPENGL_CALL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
                    kFormat, kDataType, cpu_tensor->host()));
    }

    void FBOSession::GetOutputs(TensorList outputs){
        // Only One output tensor supported
        CHECK_EQ(outputs.size(), 1);
        Tensor* device_tensor = output_tensors_[0][0];
        Download(outputs[0]);
    }
}//namespace opengl
