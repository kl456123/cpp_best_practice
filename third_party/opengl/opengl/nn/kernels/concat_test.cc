#include "opengl/nn/kernels/kernel_test_utils.h"
#include "opengl/nn/kernels/concat.h"


using namespace ::opengl::testing;

namespace opengl{
    namespace{
        inline IntList ShapeToStride(IntList shape){
            IntList stride;
            int num_elements = 1;
            for(auto item:shape){
                num_elements*=item;
            }

            int temp = num_elements;
            for(auto item:shape){
                temp/=item;
                stride.emplace_back(temp);
            }
            return stride;
        }
        inline IntList OffsetToCoord(const int index, const IntList shape){
            auto stride = ShapeToStride(shape);
            // return {index%stride};
            IntList coords;
            for(auto s:stride){
                coords.emplace_back(index%s);
            }
            return coords;
        }

        inline int CoordToOffset(const IntList coords, const IntList shape){
            auto stride = ShapeToStride(shape);
            int offset = 0;
            // return {index%stride};
            for(int i=0;i<stride.size();++i){
                offset+=stride[i]*coords[i];
            }
            return offset;
        }

        void ConcatCPU(const float* input1, const float* input2,
                const int axis, const IntList& input_shape1, const IntList& input_shape2,
                const IntList& output_shape, float* output){
            // get total num first
            int output_num_elements = 1;
            for(auto item:output_shape){
                output_num_elements*=item;
            }

            for(int i=0;i<output_num_elements;++i){
                // get its coords in output
                auto coord = OffsetToCoord(i, output_shape);
                const int index = coord[axis];
                if(i==15){
                    int a = 10;
                }
                float value;
                if(index<input_shape1[axis]){
                    // use input1
                    value = input1[i];
                }else{
                    coord[axis] = index - input_shape1[axis];
                    const int offset = CoordToOffset(coord, input_shape2);
                    value = input2[offset];
                }
                output[i] = value;
            }
        }
        const IntList shape1{1, 3, 5};
        const IntList shape2{1, 3, 5};
        const int axis = 1;

        const ::dlxnet::ModelProto BuildGraph(){
            auto scope = std::unique_ptr<Scope>(new Scope());
            auto scope_ptr = scope.get();

            int const1_id = AddConstNode(scope_ptr, "const1", shape1,
                    dlxnet::TensorProto::ANY4, dlxnet::TensorProto::ANY);
            int const2_id = AddConstNode(scope_ptr, "const2", shape2,
                    dlxnet::TensorProto::ANY4, dlxnet::TensorProto::ANY);
            AddConcatNode(scope_ptr, "output", {const1_id, const2_id}, {axis});

            // add meta data
            scope->AddOutputName("output");
            return scope->ExportModel();
        }
    }//namespace

    TEST(ConcatTest, SimpleTest){
        auto session = InitSession();
        session->LoadGraph(BuildGraph());

        // std::vector<int> image_shape = {num_inputs, input_height, input_width, input_channels};
        // ::opengl::NamedTensorList inputs(1);
        ::opengl::TensorList outputs_cpu;
        // inputs[0].first = "input";
        // inputs[0].second = Tensor::Random(Tensor::DT_FLOAT, image_shape, dlxnet::TensorProto::ANY);

        session->Setup({});

        // do computation for the graph
        session->Run();

        ::opengl::TensorNameList output_names({"output", "const1", "const2"});
        ::opengl::StringList dformats({"ANY", "ANY", "ANY"});

        // get cpu outputs from device
        session->GetOutputs(output_names, dformats, &outputs_cpu);

        // check the result
        // check the shape first
        const float* ogl_output_data = outputs_cpu[0]->host<float>();
        const int output_num_elements = outputs_cpu[0]->num_elements();
        const auto output_shape = outputs_cpu[0]->shape();
        for(int i=0;i<output_shape.size();++i){
            if(axis!=i){
                EXPECT_EQ(output_shape[i], shape1[i]);
            }else{
                EXPECT_EQ(output_shape[i], shape1[i]+shape2[i]);
            }
        }
        // check the value
        const float* ogl_const1_data = outputs_cpu[1]->host<float>();
        const float* ogl_const2_data = outputs_cpu[2]->host<float>();
        float* cpu_output_data = new float[output_num_elements];
        ConcatCPU(ogl_const1_data, ogl_const2_data, axis, outputs_cpu[1]->shape(),
                outputs_cpu[2]->shape(), outputs_cpu[0]->shape(), cpu_output_data);
        for(int i=0;i<output_num_elements;++i){
            EXPECT_EQ(ogl_output_data[i], cpu_output_data[i]);
        }
    }
}//namespace opengl
