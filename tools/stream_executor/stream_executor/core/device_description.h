#ifndef STREAM_EXECUTOR_CORE_DEVICE_DESCRIPTION_H_
#define STREAM_EXECUTOR_CORE_DEVICE_DESCRIPTION_H_
#include <string>
#include <cstdint>
#include <memory>
#include <map>
#include "stream_executor/utils/macros.h"
#include "stream_executor/core/launch_dim.h"

typedef int64_t int64;

using std::string;

namespace internal{
    class DeviceDescriptionBuilder;
}

class DeviceDescription{
    public:
        // Returns the device vendor string, e.g., "NVIDIA Corporation", "Advanced
        // Micro Devices, Inc.", or "GenuineIntel".
        const string &device_vendor() const { return device_vendor_; }
        const string& platform_version()const{return platform_version_;}
        // Returns the driver version interfacing with the underlying platform. Vendor
        // dependent format.
        const string &driver_version() const { return driver_version_; }

        bool cuda_compute_capability(int *major, int *minor) const;
        // Return the runtime version, if one is provided by the underlying platform.
        // Vendor dependent format / usefulness.
        const string &runtime_version() const { return runtime_version_; }
        // Returns the name that the device reports. Vendor dependent.
        const string &name() const { return name_; }
        // Returns the PCI bus identifier for this device, of the form
        // [domain]:[bus]:[device].[function]
        const string &pci_bus_id() const { return pci_bus_id_; }
        // Number of cores (traditional notion of core; i.e. an SM on an NVIDIA device
        // or an AMD Compute Unit.
        int core_count() const { return core_count_; }
        // Returns the limit on the thread dimensionality values in each of the
        // respective dimensions. These limits affect what constitutes a legitimate
        // kernel launch request.
        const ThreadDim &thread_dim_limit() const { return thread_dim_limit_; }

        // Returns the limit on the block dimensionality values in each of the
        // respective dimensions. These limits may affect what constitutes a
        // legitimate kernel launch request.
        const BlockDim &block_dim_limit() const { return block_dim_limit_; }

        // Returns the limit on the total number of threads that can be launched in a
        // single block; i.e. the limit on x * y * z dimensions of a ThreadDim.
        // This limit affects what constitutes a legitimate kernel launch request.
        const int64 &threads_per_block_limit() const {
            return threads_per_block_limit_;
        }

        // Returns the limit on the total number of threads that can be simultaneously
        // launched on a given multiprocessor.
        const int64 &threads_per_core_limit() const {
            return threads_per_core_limit_;
        }

        // Returns the number of threads per warp/wavefront.
        const int64 &threads_per_warp() const { return threads_per_warp_; }

        // Returns the limit on the total number of registers per core.
        const int64 &registers_per_core_limit() const {
            return registers_per_core_limit_;
        }

        // Returns the limit on the total number of registers that can be
        // simultaneously used by a block.
        const int64 &registers_per_block_limit() const {
            return registers_per_block_limit_;
        }
        // Returns the number of address bits available to kernel code running on the
        // platform. This affects things like the maximum allocation size and perhaps
        // types used in kernel code such as size_t.
        const int64 &device_address_bits() const { return device_address_bits_; }
        // Returns the device memory size in bytes.
        int64 device_memory_size() const { return device_memory_size_; }
        // Returns the device's memory bandwidth in bytes/sec.  (This is for
        // reads/writes to/from the device's own memory, not for transfers between the
        // host and device.)
        int64 memory_bandwidth() const { return memory_bandwidth_; }
        // Returns the device's core clock rate in GHz.
        float clock_rate_ghz() const { return clock_rate_ghz_; }
        // For string values that are not available via the underlying platform, this
        // value will be provided.
        static const char *kUndefinedString;

        std::unique_ptr<std::map<string, string>> ToMap() const;
    private:
        friend class internal::DeviceDescriptionBuilder;

        DeviceDescription();
        // For description of the following members, see the corresponding accessor
        // above.
        //
        // N.B. If another field is added, update ToMap() above.
        string device_vendor_;
        string platform_version_;
        string driver_version_;
        string runtime_version_;
        string pci_bus_id_;
        string name_;

        ThreadDim thread_dim_limit_;
        BlockDim block_dim_limit_;

        int64 threads_per_core_limit_;
        int64 threads_per_block_limit_;
        int64 threads_per_warp_;

        int64 registers_per_core_limit_;
        int64 registers_per_block_limit_;

        int64 device_address_bits_;
        int64 device_memory_size_;
        int64 memory_bandwidth_;

        // Shared memory limits on a given device.
        int64 shared_memory_per_core_;
        int64 shared_memory_per_block_;

        float clock_rate_ghz_;

        // CUDA "CC" major value, -1 if not available.
        int cuda_compute_capability_major_;
        int cuda_compute_capability_minor_;

        // ROCM AMDGPU ISA version, 0 if not available.
        int rocm_amdgpu_isa_version_;

        int numa_node_;
        int core_count_;
        bool ecc_enabled_;

        DISALLOW_COPY_AND_ASSIGN(DeviceDescription);

};

namespace internal{
    class DeviceDescriptionBuilder{
        public:
            DeviceDescriptionBuilder();
            // For descriptions of the following fields, see comments on the corresponding
            // DeviceDescription::* accessors above.


            void set_device_vendor(const string &value) {
                device_description_->device_vendor_ = value;
            }
            void set_platform_version(const string &value) {
                device_description_->platform_version_ = value;
            }
            void set_driver_version(const string &value) {
                device_description_->driver_version_ = value;
            }
            void set_runtime_version(const string &value) {
                device_description_->runtime_version_ = value;
            }
            void set_pci_bus_id(const string &value) {
                device_description_->pci_bus_id_ = value;
            }
            void set_name(const string &value) { device_description_->name_ = value; }

            void set_thread_dim_limit(const ThreadDim &value) {
                device_description_->thread_dim_limit_ = value;
            }
            void set_block_dim_limit(const BlockDim &value) {
                device_description_->block_dim_limit_ = value;
            }

            void set_threads_per_core_limit(int64 value) {
                device_description_->threads_per_core_limit_ = value;
            }
            void set_threads_per_block_limit(int64 value) {
                device_description_->threads_per_block_limit_ = value;
            }
            void set_threads_per_warp(int64 value) {
                device_description_->threads_per_warp_ = value;
            }

            void set_registers_per_core_limit(int64 value) {
                device_description_->registers_per_core_limit_ = value;
            }
            void set_registers_per_block_limit(int64 value) {
                device_description_->registers_per_block_limit_ = value;
            }

            void set_device_address_bits(int64 value) {
                device_description_->device_address_bits_ = value;
            }
            void set_device_memory_size(int64 value) {
                device_description_->device_memory_size_ = value;
            }
            void set_memory_bandwidth(int64 value) {
                device_description_->memory_bandwidth_ = value;
            }

            void set_shared_memory_per_core(int64 value) {
                device_description_->shared_memory_per_core_ = value;
            }
            void set_shared_memory_per_block(int64 value) {
                device_description_->shared_memory_per_block_ = value;
            }

            void set_clock_rate_ghz(float value) {
                device_description_->clock_rate_ghz_ = value;
            }

            void set_cuda_compute_capability(int major, int minor) {
                device_description_->cuda_compute_capability_major_ = major;
                device_description_->cuda_compute_capability_minor_ = minor;
            }

            void set_core_count(int value) { device_description_->core_count_ = value; }

            // Returns a built DeviceDescription with ownership transferred to the
            // caller. There are currently no restrictions on which fields must be set in
            // order to build the descriptor.
            //
            // Once the description is built, this builder object should be discarded.
            std::unique_ptr<DeviceDescription> Build() {
                return std::move(device_description_);
            }

        private:
            std::unique_ptr<DeviceDescription> device_description_;
            DISALLOW_COPY_AND_ASSIGN(DeviceDescriptionBuilder);
    };
}


#endif
