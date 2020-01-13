#ifndef CONFIG_PARSER_H_
#define CONFIG_PARSER_H_
#include <string>
#include <memory>
#include <stdio.h>
#include <sys/types.h>
#include <fcntl.h>
#include <iostream>
#include <fstream>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>



template<typename ProtoType>
class ConfigParser{
    public:
        ConfigParser();
        virtual ~ConfigParser(){
        }


        void LoadFromTxt(std::string& fn);
        void LoadFromBinary(std::string& fn);

        void SaveToTxt(std::string& fn);
        void SaveToBinary(std::string& fn);

        void Print();
        ProtoType* config_proto(){
            return config_proto_.get();
        }

    private:
        std::string filename_;
        std::shared_ptr<ProtoType> config_proto_;
};


template<typename ProtoType>
ConfigParser<ProtoType>::ConfigParser(){
    config_proto_.reset(new ProtoType());
}

template<typename ProtoType>
void ConfigParser<ProtoType>::LoadFromTxt(std::string& fn){
    int fd = open(fn.c_str(), O_RDONLY);
    if(fd<0){
        std::cout<<"error when opening demo.cfg."<<std::endl;
        return;
    }
    google::protobuf::io::FileInputStream input(fd);
    input.SetCloseOnDelete(true);
    google::protobuf::TextFormat::Parse(&input, config_proto_.get());
}


template<typename ProtoType>
void ConfigParser<ProtoType>::LoadFromBinary(std::string& fn){
    std::fstream input(filename_, std::ios::in |std::ios::binary);
    config_proto_->ParseFromIstream(&input);
}

template<typename ProtoType>
void ConfigParser<ProtoType>::SaveToTxt(std::string& fn){
    std::string str_proto;
    google::protobuf::TextFormat::PrintToString(*config_proto_, &str_proto);
    std::fstream output(fn, std::ios::out|std::ios_base::ate);
    if(!output){
        std::cout<<"error during saving to txt."<<std::endl;
        return;
    }
    output<<str_proto<<std::endl;
    output.flush();
    output.close();
}

template<typename ProtoType>
void ConfigParser<ProtoType>::SaveToBinary(std::string& fn){
    std::fstream output(fn, std::ios::out | std::ios::trunc | std::ios::binary);
    config_proto_->SerializeToOstream(&output);
}


template<typename ProtoType>
void ConfigParser<ProtoType>::Print(){
    config_proto_->PrintDebugString();
}


#endif
