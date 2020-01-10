#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <sys/types.h>
#include <fcntl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

using namespace std;

#include "search_request.pb.h"


// fill it
void PromptForAddress(SearchRequest* search_req){
    search_req->set_query("dga");
    search_req->set_page_number(10);
    search_req->set_result_per_page(10);
    search_req->set_corpus(SearchRequest_Corpus_UNIVERSAL);
}

void ListProto(SearchRequest* search_req){
    cout<<"query: "<<search_req->query()<<endl;
    cout<<"number: "<<search_req->page_number()<<endl;
    cout<<"result_per_page: "<<search_req->result_per_page();
}

void SaveToTxt(SearchRequest* search_req){
    string str_proto;
    google::protobuf::TextFormat::PrintToString(*search_req, &str_proto);
    fstream output("demo.cfg", ios::out|ios_base::ate);
    if(!output){
        cout<<"error during saving to txt."<<endl;
        return;
    }
    output<<str_proto<<endl;
    output.flush();
    output.close();

}

void LoadFromTxt(SearchRequest* search_req){
    int fd = open("demo.cfg", O_RDONLY);
    if(fd<0){
        cout<<"error when opening demo.cfg."<<endl;
        return;
    }
    google::protobuf::io::FileInputStream input(fd);
    input.SetCloseOnDelete(true);
    google::protobuf::TextFormat::Parse(&input, search_req);
    search_req->PrintDebugString();
}



int main(int argc, char* argv[]){
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    SearchRequest search_req;
    if(argc!=2){
        cerr<<"Usage: "<<argv[0]<< "search request file"<<endl;
        return -1;
    }
    fstream input(argv[1], ios::in |ios::binary);
    if(!input){
        cout<<argv[1]<<": File not found. Create a new file."<<endl;
        PromptForAddress(&search_req);
        fstream output(argv[1], ios::out | ios::trunc | ios::binary);
        if(!search_req.SerializeToOstream(&output)){
            SaveToTxt(&search_req);
            cerr<<"Failed to write search request. "<<endl;
            return -1;
        }
    }else{
        search_req.ParseFromIstream(&input);
        LoadFromTxt(&search_req);
    }

    ListProto(&search_req);

    google::protobuf::ShutdownProtobufLibrary();

    return 0;
}
