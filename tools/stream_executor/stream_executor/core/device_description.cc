#include <memory>
#include <map>

#include "stream_executor/utils/strcat.h"
#include "stream_executor/core/device_description.h"


static const uint64 kUninitializedUint64 = -1ULL;
/* static */ const char *DeviceDescription::kUndefinedString = "<undefined>";

DeviceDescription::DeviceDescription()
    : device_vendor_(kUndefinedString),
    platform_version_(kUndefinedString),
    driver_version_(kUndefinedString),
    runtime_version_(kUndefinedString),
    pci_bus_id_(kUndefinedString),
    name_(kUndefinedString),
    thread_dim_limit_(kUninitializedUint64, kUninitializedUint64,
            kUninitializedUint64),
    block_dim_limit_(kUninitializedUint64, kUninitializedUint64,
            kUninitializedUint64),
    threads_per_core_limit_(kUninitializedUint64),
    threads_per_block_limit_(kUninitializedUint64),
    threads_per_warp_(kUninitializedUint64),
    registers_per_core_limit_(kUninitializedUint64),
    registers_per_block_limit_(kUninitializedUint64),
    device_address_bits_(kUninitializedUint64),
    device_memory_size_(kUninitializedUint64),
    memory_bandwidth_(kUninitializedUint64),
    shared_memory_per_core_(kUninitializedUint64),
    shared_memory_per_block_(kUninitializedUint64),
    clock_rate_ghz_(-1.0),
    cuda_compute_capability_major_(-1),
    cuda_compute_capability_minor_(-1),
    rocm_amdgpu_isa_version_(-1),
    numa_node_(-1),
    core_count_(-1),
    ecc_enabled_(false) {}


    std::unique_ptr<std::map<string, string>> DeviceDescription::ToMap() const {
        std::unique_ptr<std::map<string, string>> owned_result{
            new std::map<string, string>};
        std::map<string, string> &result = *owned_result;
        result["Device Vendor"] = device_vendor();
        result["Platform Version"] = platform_version();
        result["Driver Version"] = driver_version();
        result["Runtime Version"] = runtime_version();
        result["PCI bus ID"] = pci_bus_id_;
        result["Device Name"] = name_;

        const ThreadDim &thread_dim = thread_dim_limit();
        result["ThreadDim Limit"] =
            string_utils::str_cat(std::to_string(thread_dim.x), ",",
                    std::to_string(thread_dim.y), ",",
                    std::to_string(thread_dim.z));
        const BlockDim &block_dim = block_dim_limit();
        result["BlockDim Limit"] =
            string_utils::str_cat(std::to_string(block_dim.x), ",",
                    std::to_string(block_dim.y), ",",
                    std::to_string(block_dim.z));

        result["Threads Per Core Limit"] = std::to_string(threads_per_core_limit());
        result["Threads Per Block Limit"] = std::to_string(threads_per_block_limit());
        result["Registers Per Block Limit"] =
            std::to_string(registers_per_block_limit());

        result["Device Address Bits"] = std::to_string(device_address_bits());
        result["Device Memory Size"] =
            std::to_string(device_memory_size());
        result["Memory Bandwidth"] = string_utils::str_cat(
                std::to_string(memory_bandwidth_), "/s");

        result["Shared Memory Per Core"] =
            std::to_string(shared_memory_per_core_);
        result["Shared Memory Per Block"] =
            std::to_string(shared_memory_per_block_);

        result["Clock Rate GHz"] = std::to_string(clock_rate_ghz());

        result["CUDA Compute Capability"] = string_utils::str_cat(
                std::to_string(cuda_compute_capability_major_), ".",
                std::to_string(cuda_compute_capability_minor_));

        // result["NUMA Node"] = std::to_string(numa_node());
        result["Core Count"] = std::to_string(core_count());
        // result["ECC Enabled"] = std::to_string(ecc_enabled());
        return owned_result;
    }

namespace internal{
    DeviceDescriptionBuilder::DeviceDescriptionBuilder():device_description_(new DeviceDescription){}
}

bool DeviceDescription::cuda_compute_capability(int *major, int *minor) const {
    *major = cuda_compute_capability_major_;
    *minor = cuda_compute_capability_minor_;
    return cuda_compute_capability_major_ != 0;
}
