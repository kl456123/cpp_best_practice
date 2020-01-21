#ifndef CORE_OP_BUILD_H_
#define CORE_OP_BUILD_H_
#include <string>
#include <functional>
#include <vector>

#include "core/error.hpp"
#include "op_def.pb.h"
#include "shape_inference.h"

/**
 * builder of operator to assign param to op more convenient
 * call it by chains style
 * */

// param used for register operator
typedef std::function<Status(int)> OpShapeInferenceFn;
struct OpRegistrationData{
    public:
        OpRegistrationData(){}
        OpRegistrationData(const OpDef& def):op_def(def){}
        OpRegistrationData(const OpDef& def, const OpShapeInferenceFn& fn)
            :op_def(def),shape_inference_fn(fn){}
        // params
        OpDef op_def;
        OpShapeInferenceFn shape_inference_fn;


};
// used for decorate op_reg_data
class OpDefBuilder{
    public:
        explicit OpDefBuilder(std::string name);
        //no need to release memory
        virtual ~OpDefBuilder(){}

        OpDefBuilder& Input(std::string spec);
        OpDefBuilder& Output(std::string spec);
        OpDefBuilder& SetShapeFn(OpShapeInferenceFn fn);

        // add attrs
        OpDefBuilder& Attr(std::string spec);

        Status Finalize(OpRegistrationData* op_reg_data)const;

    private:
        // used for convenience
        // return proto type
        OpDef* op_def(){return &op_reg_data_.op_def;}
        OpRegistrationData op_reg_data_;

        // attrs, inputs and outputs
        std::vector<std::string> attrs_;
        std::vector<std::string> inputs_;
        std::vector<std::string> outputs_;

        // collect all errors
        std::vector<std::string> errors_;
};


#endif
