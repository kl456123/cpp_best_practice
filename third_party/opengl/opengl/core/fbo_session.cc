#include "opengl/core/fbo_session.h"
#include "opengl/utils/macros.h"
#include "opengl/core/kernel.h"
#include "opengl/core/kernel_registry.h"
#include "opengl/utils/protobuf.h"
#include "opengl/core/tensor.h"


namespace opengl{
    namespace{
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

    FBOSession::~FBOSession(){
        glDeleteFramebuffers(1, &frame_buffer_);
    }

    void FBOSession::SetupFrameBuffer(){

        OPENGL_CALL(glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer_));
        // Set the list of draw buffers.
        GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
        // "1" is the size of DrawBuffers.
        OPENGL_CALL(glDrawBuffers(1, DrawBuffers));
    }

    void FBOSession::LoadGraph(const std::string file_path){
        // load graph from disk
        CHECK(ReadProtoFromBinary(file_path.c_str(), model_))
            <<"Load Graph "<<file_path <<"Failed";

        dlxnet::GraphProto graph = model_->graph();
        // create kernel and setup input and output for each node
        // Note that dont need to allocate memory due to lack of shape information
        total_tensors_.resize(graph.tensor_names_size());


        // build tensor_name -> tensor_index map
        for(int i=0;i<graph.tensor_names_size();++i){
            tensor_name_index_[graph.tensor_names(i)] = i;
        }


        Kernel* kernel;
        for(auto& node: graph.node()){
            kernel=nullptr;
            KernelRegistry::Global()->CreateKernel(node.type(), &kernel, context_);
            if(kernel==nullptr){
                LOG(FATAL)<<"unsupported kernel name "<<node.type();
            }
            // setup program for each kernel here
            kernel->SetupProgram(vertex_shader_);

            kernel->SetupAttr(node.attr());
            // fill inputs and outputs
            for(int i=0;i<node.input_index_size();++i){
                kernel->input_tensor_indexes_.emplace_back(node.input_index(i));
            }
            for(int i=0;i<node.output_index_size();++i){
                kernel->output_tensor_indexes_.emplace_back(node.output_index(i));
            }
            kernels_.emplace_back(kernel);
        }
        OPENGL_CHECK_ERROR;
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
            Kernel* kernel = kernels_[i];

            // clear first
            kernel->input_tensors_.clear();
            kernel->output_tensors_.clear();

            // prepare input and output
            for(auto& index:kernel->input_tensor_indexes_){
                kernel->input_tensors_.emplace_back(total_tensors_[index]);
            }

            for(auto&index:kernel->output_tensor_indexes_){
                kernel->output_tensors_.emplace_back(total_tensors_[index]);
            }
            kernel->Compute();
            OPENGL_CHECK_ERROR;
        }
    }


    FBOSession::FBOSession(Context* context)
        :context_(context){
            // create vertex shader first
            CreateVertexShader();

            model_ = new dlxnet::ModelProto;

            // only need to create it once
            OPENGL_CALL(glGenFramebuffers(1, &frame_buffer_));
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

    void FBOSession::Setup(const NamedTensorList& inputs_cpu){
        // allocate memory for each tensor
        // so that dont need to allocate input and output tensors
        // for each kernel during computation
        // set up global framebuffer, all nodes are only needed to
        // attach output texture to the global frame buffer
        SetupFrameBuffer();

        // allocate memory for input tensor(device_tensor) first
        // TODO(breakpoint) add input-typed kernel
        for(auto input_iter=inputs_cpu.begin();input_iter!=inputs_cpu.end();++input_iter){
            const Tensor* input_cpu = input_iter->second;
            const auto& tensor_name = input_iter->first;

            auto iter = tensor_name_index_.find(tensor_name);
            if(iter==tensor_name_index_.end()){
                LOG(FATAL)<<"tensor_name: "<<tensor_name<<"Cannot Find";
            }
            const int input_index = iter->second;

            // allocate memory
            total_tensors_[input_index] = new Tensor(Tensor::DT_FLOAT, input_cpu->shape(),
                    Tensor::DEVICE_TEXTURE, dlxnet::TensorProto::NHWC4);

            // upload data, initialize input tensor
            context_->CopyCPUTensorToDevice(input_cpu, total_tensors_[input_index]);
        }
        for(int i=0;i<kernels_.size();++i){
            Kernel* kernel = kernels_[i];
            TensorShapeList input_shapes, output_shapes;
            // prepare input_shapes
            for(int j=0; j<kernel->input_tensor_indexes_.size(); ++j){
                Tensor* input_tensor = total_tensors_[kernel->input_tensor_indexes_[j]];
                CHECK(input_tensor)<<"input tensor is uninitialized of kernel index: "<<i;
                input_shapes.emplace_back(input_tensor->shape());
            }

            // infer output shapes from input shapes
            kernel->InferOutputShape(input_shapes, output_shapes);

            // allocate memory for each output tensors according to their shapes
            for(int j=0;j<output_shapes.size();++j){
                auto dformat = kernel->GetOutputDFormat(j);
                total_tensors_[kernel->output_tensor_indexes_[j]] =
                    new Tensor(Tensor::DT_FLOAT, output_shapes[j],
                            Tensor::DEVICE_TEXTURE, dformat);
            }
        }
        OPENGL_CHECK_ERROR;
    }



    void FBOSession::GetOutputs(const TensorNameList& output_names,
            const StringList& output_dformats, TensorList* outputs){
        CHECK_EQ(output_names.size(), output_dformats.size());
        outputs->clear();
        outputs->reserve(output_names.size());

        int index = 0;
        for(auto& tensor_name: output_names){
            auto iter = tensor_name_index_.find(tensor_name);
            if(iter==tensor_name_index_.end()){
                LOG(FATAL)<<"tensor_name: "<<tensor_name<<"Cannot Find";
            }

            const int tensor_index = tensor_name_index_[tensor_name];
            const Tensor* gpu_tensor = total_tensors_[tensor_index];
            auto dformat_str = output_dformats[index++];
            DataFormat dformat;
            if(dformat_str=="NHWC"){
                dformat = dlxnet::TensorProto::NHWC;
            }else if(dformat_str=="NCHW"){
                dformat = dlxnet::TensorProto::NCHW;
            }else{
                LOG(FATAL)<<"only nhwc and nchw dformats are supported for now";
            }
            Tensor* cpu_tensor = new Tensor(Tensor::DT_FLOAT, gpu_tensor->shape(),
                    Tensor::HOST_MEMORY, dformat);
            context_->CopyDeviceTensorToCPU(gpu_tensor, cpu_tensor);
            outputs->emplace_back(cpu_tensor);
        }
    }

    std::string FBOSession::DebugString()const{
        std::string ret_str;
        ret_str+="ModelProto: ";
        ret_str+=model_->DebugString();
        ret_str+="\n";
        return ret_str;
    }
}//namespace opengl
