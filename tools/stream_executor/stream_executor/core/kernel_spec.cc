#include "stream_executor/core/kernel_spec.h"



KernelLoaderSpec::KernelLoaderSpec(string kernelname)
    : kernelname_(string(kernelname)) {}

OnDiskKernelLoaderSpec::OnDiskKernelLoaderSpec(string filename,
        string kernelname)
    : KernelLoaderSpec(kernelname), filename_(string(filename)) {}


CudaPtxOnDisk::CudaPtxOnDisk(string filename,
        string kernelname)
    : OnDiskKernelLoaderSpec(filename, kernelname) {}

CudaCubinInMemory::CudaCubinInMemory(const char *bytes,
        string kernelname)
    : KernelLoaderSpec(kernelname), bytes_(bytes) {}

CudaCubinOnDisk::CudaCubinOnDisk(string filename,
        string kernelname)
    : OnDiskKernelLoaderSpec(filename, kernelname) {}

    const std::tuple<int, int> CudaPtxInMemory::kMinimumCapability{1, 0};




bool CompareComputeCapability(const std::tuple<int, int> &lhs,
        const std::tuple<int, int> &rhs) {
    return std::get<0>(lhs) < std::get<0>(rhs) ||
        (std::get<0>(lhs) == std::get<0>(rhs) &&
         std::get<1>(lhs) < std::get<1>(rhs));
}


CudaPtxInMemory::CudaPtxInMemory(std::string ptx,
        std::string kernel_name,
        bool ptx_compressed)
    : KernelLoaderSpec(kernel_name),
    ptx_by_compute_capability_(CompareComputeCapability) {
        if (ptx_compressed) {
            // Lazy decompression. Put an empty string in decompressed_ptx_ showing that
            // the original ptx is compressed.
            decompressed_ptx_[ptx.data()] = "";
        }
        ptx_by_compute_capability_[kMinimumCapability] = ptx.data();
    }


CudaPtxInMemory::CudaPtxInMemory(
        const std::initializer_list<CudaPtxInMemory::PtxSpec> &spec_list,
        string kernel_name, bool ptx_compressed)
    : KernelLoaderSpec(kernel_name),
    ptx_by_compute_capability_(CompareComputeCapability) {
        for (const auto &spec : spec_list) {
            int major, minor;
            string ptx;
            std::tie(major, minor, ptx) = spec;
            if (ptx_compressed) {
                // Lazy decompression. Put an empty string in decompressed_ptx_ showing
                // that the original ptx is compressed.
                decompressed_ptx_[ptx.data()] = "";
            }
            ptx_by_compute_capability_[std::tuple<int, int>{major, minor}] = ptx.data();
        }
    }

string CudaPtxInMemory::DecompressPtx(const char *ptx) {
    // Get the length of the PTX string from the beginning of the buffer.
    uint64 ptx_length = *reinterpret_cast<const uint64 *>(ptx);
    // Get the PTX string from the buffer with offset and length.
    string compressed_ptx(ptx + sizeof(uint64),
            ptx + sizeof(uint64) + ptx_length);
    string decompressed_ptx;
    // Decompress the PTX string with bzip2.
    LOG(FATAL) << "bzip2 decompression is not supported yet.";
    return decompressed_ptx;
}

const char *CudaPtxInMemory::default_text() const {
    if (ptx_by_compute_capability_.empty()) {
        return nullptr;
    }

    auto ptx = ptx_by_compute_capability_.begin()->second;
    // Check if there is an entry in decompressed ptx table.
    auto decompressed_ptx_iter = decompressed_ptx_.find(ptx);
    if (decompressed_ptx_iter != decompressed_ptx_.end()) {
        // If the decompressed string is empty, which means the ptx hasn't been
        // decompressed, decompress it here.
        if (decompressed_ptx_iter->second.empty()) {
            decompressed_ptx_iter->second = DecompressPtx(ptx);
        }
        return decompressed_ptx_iter->second.c_str();
    }
    return ptx;
}

const char *CudaPtxInMemory::original_default_text() const {
    if (ptx_by_compute_capability_.empty()) {
        return nullptr;
    }

    return ptx_by_compute_capability_.begin()->second;
}

const char *CudaPtxInMemory::text(int compute_capability_major,
        int compute_capability_minor) const {
    std::tuple<int, int> capability{compute_capability_major,
        compute_capability_minor};

    auto ptx_iter = ptx_by_compute_capability_.find(capability);
    if (ptx_iter == ptx_by_compute_capability_.end()) {
        return nullptr;
    }

    // Check if there is an entry in decompressed ptx table.
    auto decompressed_ptx_iter = decompressed_ptx_.find(ptx_iter->second);
    if (decompressed_ptx_iter != decompressed_ptx_.end()) {
        // If the decompressed string is empty, which means the ptx hasn't been
        // decompressed, decompress it here.
        if (decompressed_ptx_iter->second.empty()) {
            decompressed_ptx_iter->second = DecompressPtx(ptx_iter->second);
        }
        return decompressed_ptx_iter->second.c_str();
    }
    return ptx_iter->second;
}

const char *CudaPtxInMemory::original_text(int compute_capability_major,
        int compute_capability_minor) const {
    std::tuple<int, int> capability{compute_capability_major,
        compute_capability_minor};

    auto ptx_iter = ptx_by_compute_capability_.find(capability);
    if (ptx_iter == ptx_by_compute_capability_.end()) {
        return nullptr;
    }

    return ptx_iter->second;
}

OpenCLTextOnDisk::OpenCLTextOnDisk(string filename,
        string kernelname)
    : OnDiskKernelLoaderSpec(filename, kernelname) {}

OpenCLTextInMemory::OpenCLTextInMemory(string text,
        string kernelname)
    : KernelLoaderSpec(kernelname), text_(text) {}

OpenCLBinaryOnDisk::OpenCLBinaryOnDisk(string filename,
        string kernelname)
    : OnDiskKernelLoaderSpec(filename, kernelname) {}

    MultiKernelLoaderSpec *MultiKernelLoaderSpec::AddOpenCLTextOnDisk(
            string filename, string kernelname) {
        CHECK(ocl_text_on_disk_ == nullptr);
        ocl_text_on_disk_.reset(new OpenCLTextOnDisk{filename, kernelname});
        return this;
    }

MultiKernelLoaderSpec *MultiKernelLoaderSpec::AddOpenCLBinaryOnDisk(
        string filename, string kernelname) {
    CHECK(ocl_binary_on_disk_ == nullptr);
    ocl_binary_on_disk_.reset(new OpenCLBinaryOnDisk{filename, kernelname});
    return this;
}

MultiKernelLoaderSpec *MultiKernelLoaderSpec::AddOpenCLTextInMemory(
        string filename, string kernelname) {
    CHECK(ocl_text_in_memory_ == nullptr);
    ocl_text_in_memory_.reset(new OpenCLTextInMemory{filename, kernelname});
    return this;
}

MultiKernelLoaderSpec *MultiKernelLoaderSpec::AddCudaPtxOnDisk(
        string filename, string kernelname) {
    CHECK(cuda_ptx_on_disk_ == nullptr);
    cuda_ptx_on_disk_.reset(new CudaPtxOnDisk{filename, kernelname});
    return this;
}

MultiKernelLoaderSpec *MultiKernelLoaderSpec::AddCudaCubinInMemory(
        const char *bytes, string kernelname) {
    CHECK(cuda_cubin_in_memory_ == nullptr);
    cuda_cubin_in_memory_.reset(new CudaCubinInMemory{bytes, kernelname});
    return this;
}

MultiKernelLoaderSpec *MultiKernelLoaderSpec::AddCudaCubinOnDisk(
        string filename, string kernelname) {
    CHECK(cuda_cubin_on_disk_ == nullptr);
    cuda_cubin_on_disk_.reset(new CudaCubinOnDisk{filename, kernelname});
    return this;
}

MultiKernelLoaderSpec *MultiKernelLoaderSpec::AddCudaPtxInMemory(
        string ptx, string kernelname) {
    CHECK(cuda_ptx_in_memory_==nullptr);
    cuda_ptx_in_memory_.reset(
            new CudaPtxInMemory{ptx, kernelname, false /* ptx_compressed */});
    return this;
}

MultiKernelLoaderSpec *MultiKernelLoaderSpec::AddCudaCompressedPtxInMemory(
        string ptx, string kernelname) {
    CHECK(cuda_ptx_in_memory_ == nullptr);
    cuda_ptx_in_memory_.reset(
            new CudaPtxInMemory{ptx, kernelname, true /* ptx_compressed */});
    return this;
}

MultiKernelLoaderSpec *MultiKernelLoaderSpec::AddCudaPtxInMemory(
        std::initializer_list<CudaPtxInMemory::PtxSpec> spec_list,
        string kernelname) {
    CHECK(cuda_ptx_in_memory_ == nullptr);
    cuda_ptx_in_memory_.reset(
            new CudaPtxInMemory{spec_list, kernelname, false /* ptx_compressed */});
    return this;
}

MultiKernelLoaderSpec *MultiKernelLoaderSpec::AddCudaCompressedPtxInMemory(
        std::initializer_list<CudaPtxInMemory::PtxSpec> spec_list,
        string kernelname) {
    CHECK(cuda_ptx_in_memory_ == nullptr);
    cuda_ptx_in_memory_.reset(
            new CudaPtxInMemory{spec_list, kernelname, true /* ptx_compressed */});
    return this;
}

MultiKernelLoaderSpec::MultiKernelLoaderSpec(size_t arity) : arity_(arity) {}

