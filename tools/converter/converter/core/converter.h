#ifndef CONVERTER_CORE_CONVERTER_H_
#define CONVERTER_CORE_CONVERTER_H_
#include <string>
#include "core/config.h"
#include "core/registry.h"
#include "dlcl.pb.h"


class Converter: public RegistryItemBase{
    public:
        Converter(){};
        void Reset(const ConverterConfig config){
            converter_config_ = config;
        }
        Converter(const ConverterConfig config);
        virtual ~Converter(){};
        // TODO(breakpoint) change to return Status
        virtual void Run()=0;

        void Save(std::string checkpoint_path);

    protected:
        ConverterConfig converter_config_;
        Model* model_;
};


// INSTANIZE_REGISTRY(Converter);
#define REGISTER_CLASS_CONVERTER(CLASS)  \
    REGISTER_CLASS(Converter, CLASS)

#endif

