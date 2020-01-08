#include <iostream>
#include <fstream>
#include <string>

#include "search_request.pb.h"


// fill it
void PromptForAddress(SearchRequest* search_req){
    search_req->set_query("dga");
    search_req->set_page_number(10);
    search_req->set_result_per_page(10);
    search_req->set_corpus(SearchRequest_Corpus_UNIVERSAL);
}



int main(){
    return 0;
}
