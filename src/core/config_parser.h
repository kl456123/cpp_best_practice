#ifndef CONFIG_PARSER_H_
#define CONFIG_PARSER_H_
#include <string>
#include <memory>



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

    private:
        std::string filename_;
        std::shared_ptr<ProtoType> config_proto_;
};


#endif
