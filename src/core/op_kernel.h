#ifndef CORE_OP_KERNEL_H_
#define CORE_OP_KERNEL_H_
#include <memory>
#include <unordered_map>
#include "core/device.h"
#include "node_def.pb.h"

/*
 * put Licensed here
 * */

class OpKernelContext;
class OpKernelConstruction;

class OpKernel{
    public:
        explicit OpKernel(OpKernelConstruction* context);
        virtual ~OpKernel();


        virtual void Compute(OpKernelContext* context)=0;


        //Accessor.
        const NodeDef& def()const{return def_;}
        const string& name()const;
        const string& requested_device()const;

        int num_inputs(){return input_types_.size();}
        int num_outputs(){return output_types_.size();}

    private:
        const std::unique_ptr<const NodeDef> def_;

        // memory type and data type for inputs and outputs
        const MemoryTypeVector input_memory_types_;
        const DataTypeVector input_types_;
        const MemoryTypeVector input_memory_types_;
        const DataTypeVector input_types_;

        std::ordered_map<std::string, int> input_name_map_;
        std::ordered_map<std::string, int> output_name_map_;
};


// allocate some memory when create op kernel
class OpKernelConstruction{
    public:


    private:
        // device and memory management
        const DeviceType device_type_;
        Device* const  device_;
        Allocator* allocator_;

        const NodeDef* def_;
        const OpDef* op_def_;

        friend class OpKernel;
};

// allocate some memory when run op kernel
class OpKernelContext{
    public:
        // used to initialize an OpKernelContext
        struct Params{
            OpKernel* op_kernel=nullptr;

            DeviceBase* device=nullptr;
        }

};


#endif
