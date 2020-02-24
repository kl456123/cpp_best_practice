#include <string>
#include <stdio.h>
#include <assert.h>

#include "stream_executor/utils/status.h"

std::ostream& operator<<(std::ostream& os, const Status& x) {
    os << x.ToString();
    return os;
}

void Status::IgnoreError() const {
    // no-op
}
std::string error_name(error::Code code) {
    switch (code) {
        case error::OK:
            return "OK";
            break;
        case error::CANCELLED:
            return "Cancelled";
            break;
        case error::UNKNOWN:
            return "Unknown";
            break;
        case error::INVALID_ARGUMENT:
            return "Invalid argument";
            break;
        case error::DEADLINE_EXCEEDED:
            return "Deadline exceeded";
            break;
        case error::NOT_FOUND:
            return "Not found";
            break;
        case error::ALREADY_EXISTS:
            return "Already exists";
            break;
        case error::PERMISSION_DENIED:
            return "Permission denied";
            break;
        case error::UNAUTHENTICATED:
            return "Unauthenticated";
            break;
        case error::RESOURCE_EXHAUSTED:
            return "Resource exhausted";
            break;
        case error::FAILED_PRECONDITION:
            return "Failed precondition";
            break;
        case error::ABORTED:
            return "Aborted";
            break;
        case error::OUT_OF_RANGE:
            return "Out of range";
            break;
        case error::UNIMPLEMENTED:
            return "Unimplemented";
            break;
        case error::INTERNAL:
            return "Internal";
            break;
        case error::UNAVAILABLE:
            return "Unavailable";
            break;
        case error::DATA_LOSS:
            return "Data loss";
            break;
        default:
            char tmp[30];
            snprintf(tmp, sizeof(tmp), "Unknown code(%d)", static_cast<int>(code));
            return tmp;
            break;
    }
}

const string& Status::empty_string() {
    static string* empty = new string;
    return *empty;
}

void Status::SlowCopyFrom(const State* src) {
    if (src == nullptr) {
        state_ = nullptr;
    } else {
        state_ = std::unique_ptr<State>(new State(*src));
    }
}
void Status::Update(const Status& new_status) {
    if (ok()) {
        *this = new_status;
    }
}

Status::Status(error::Code code, std::string msg) {
    assert(code != error::OK);
    state_ = std::unique_ptr<State>(new State);
    state_->code = code;
    state_->msg = string(msg);
}


std::string Status::ToString() const {
    if (state_ == nullptr) {
        return "OK";
    } else {
        string result(error_name(code()));
        result += ": ";
        result += state_->msg;
        return result;
    }
}

std::string* TfCheckOpHelperOutOfLine(const Status& v, const char* msg){
    std::string r("Non-OK-status: ");
    r += msg;
    r += " status: ";
    r += v.ToString();
    // Leaks string but this is only to be used in a fatal error message
    return new string(r);
}
