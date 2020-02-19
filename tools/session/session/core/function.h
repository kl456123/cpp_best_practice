#ifndef SESSION_CORE_FUNCTION_H_
#define SESSION_CORE_FUNCTION_H_
#include <cstdint>
#include <cstddef>
#include <vector>

#include "session/utils/status.h"
#include "session/core/tensor.h"
#include "session/utils/macros.h"
#include "session/core/op.h"

#include "types.pb.h"

class CallFrameInterface{
    public:
        virtual ~CallFrameInterface(){}
        virtual size_t num_args()const=0;
        virtual size_t num_retvals()const=0;
        virtual Status GetArg(int index, Tensor* val)const=0;
        virtual Status SetRetval(int index, const Tensor& val) = 0;
};


class FunctionCallFrame:public CallFrameInterface{
    public:
        FunctionCallFrame(std::vector<DataType> arg_types,
                std::vector<DataType> ret_types);
        ~FunctionCallFrame()override;
        // Caller methods.
        Status SetArgs(std::vector<Tensor> args);
        Status GetRetvals(std::vector<Tensor>* rets) const;

        // Moves the return values from the frame to rets. If allow_dead_tensors is
        // false it will fail if any of the retvals do not have a value.
        Status ConsumeRetvals(std::vector<Tensor>* rets, bool allow_dead_tensors);

        size_t num_args() const override { return arg_types_.size(); }
        size_t num_retvals() const override { return ret_types_.size(); }

        // Callee methods.
        Status GetArg(int index, Tensor* val) const override;
        Status SetRetval(int index, const Tensor& val) override;
    private:
        std::vector<DataType> arg_types_;
        std::vector<DataType> ret_types_;
        std::vector<Tensor> args_;
        struct Retval {
            bool has_val = false;
            Tensor val;
        };
        std::vector<Retval> rets_;

        DISALLOW_COPY_AND_ASSIGN(FunctionCallFrame);
};

class FunctionLibraryDefinition : public OpRegistryInterface{
    public:
          // Ops created for function arguments bear the name given by `kArgOp`; those
            // created for return values bear the name given by `kRetOp`.
        static constexpr const char* const kArgOp = "_Arg";
        ~FunctionLibraryDefinition() override;
};

#endif
