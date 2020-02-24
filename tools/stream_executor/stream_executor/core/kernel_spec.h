#ifndef STREAM_EXECUTOR_CORE_KERNEL_SPEC_H_
#define STREAM_EXECUTOR_CORE_KERNEL_SPEC_H_
#include <string>
#include <memory>
#include <map>
#include <cstdint>

#include "stream_executor/utils/logging.h"
#include "stream_executor/utils/macros.h"


using std::string;
typedef uint64_t uint64;


// Describes how to load a kernel on a target platform.
//
// This is an abstract base class, subclassed for specific platforms.
// The filename_or_text field represents the program location (i.e. PTX or
// OpenCL loadable translation unit path) and is simply stored; whether it is a
// filename or text is exposed via more specifically named accessors in
// subclasses.
//
// These kernel loader specifications are typically auto-generated into header
// files at build time, but can also be specified manually.
class KernelLoaderSpec {
    public:
        virtual ~KernelLoaderSpec() {}

        // Returns the kernel name to load out of the program.
        const string &kernelname() const { return kernelname_; }

    protected:
        explicit KernelLoaderSpec(string kernelname);

    private:
        // The kernel name that should be loaded out of the program description given
        // above.
        string kernelname_;

        DISALLOW_COPY_AND_ASSIGN(KernelLoaderSpec);
};


// An abstract kernel loader spec that has an associated file path, where
// there's a canonical suffix for the filename; e.g. see CudaPtxOnDisk whose
// canonical filename suffix is ".ptx".
class OnDiskKernelLoaderSpec : public KernelLoaderSpec {
 public:
  ~OnDiskKernelLoaderSpec() override {}

  // Returns the path to the on-disk loadable kernel file.
  const string &filename() const { return filename_; }

  // Returns the canonical suffix for this on-disk kernel loader spec format;
  // e.g. PTX files on disk have a canonical suffix of ".ptx".
  virtual const char *CanonicalSuffix() const = 0;

 protected:
  OnDiskKernelLoaderSpec(string filename,
                         string kernelname);

  string filename_;

 private:
  DISALLOW_COPY_AND_ASSIGN(OnDiskKernelLoaderSpec);
};


// Kernel loader specification for PTX text that resides on disk.
class CudaPtxOnDisk : public OnDiskKernelLoaderSpec {
 public:
  CudaPtxOnDisk(string filename, string kernelname);
  ~CudaPtxOnDisk() override {}

  const char *CanonicalSuffix() const override { return ".ptx"; }

 private:
  DISALLOW_COPY_AND_ASSIGN(CudaPtxOnDisk);
};

// Kernel loader specification for CUBIN binary that resides on disk.
class CudaCubinOnDisk : public OnDiskKernelLoaderSpec {
 public:
  CudaCubinOnDisk(string filename, string kernelname);
  ~CudaCubinOnDisk() override {}

  const string &filename() const { return filename_; }

  const char *CanonicalSuffix() const override { return ".cubin"; }

 private:
  string filename_;

  DISALLOW_COPY_AND_ASSIGN(CudaCubinOnDisk);
};

// Kernel loader specification for PTX text that resides in memory.
class CudaPtxInMemory : public KernelLoaderSpec {
 public:
  // Components: compute capability major number, compute capability minor
  // number, and PTX source.
  typedef std::tuple<int, int, string> PtxSpec;

  // Single-PTX constructor. Adds the provided PTX version with an unknown
  // compute capability. Since the CC is unknown, the PTX is assumed to be very
  // generally usable - in other words, PTX specified in this manner is VERY
  // likely to be used as the default! Note that the PTX can be compressed,
  // which is indicated by the argument ptx_compressed.
  //
  // Warning: the string backing the provided absl::string_view ptx must outlive
  // this instance.
  CudaPtxInMemory(string ptx, string kernelname,
                  bool ptx_compressed = false);

  // Multiple-PTX-version constructor. Adds each item in spec_list to this
  // object. Note that the PTX can be compressed, which is indicated by the
  // argument ptx_compressed.
  CudaPtxInMemory(const std::initializer_list<PtxSpec> &spec_list,
                  string kernel_name, bool ptx_compressed = false);
  ~CudaPtxInMemory() override {}

  // Add the PTX implementation described by ptx_spec to this object. On
  // collision (i.e., if a version with the same compute_capability already
  // exists), the existing implementation will be overwritten.
  void AddSpec(PtxSpec ptx_spec);

  // Returns pointer to the ptx of available implementation with the
  // lowest-valued compute capability. For example, if PTX written to CC2.0,
  // 3.0, and 3.5 are all available, the version for CC2.0 will be set. Returns
  // nullptr on failed lookup (if any version is not available).
  // When the ptx is compressed, returns the decompressed ptx.
  const char *default_text() const;

  // Similar to default_text().
  // When the ptx is compressed, returns the decompressed ptx.
  const char *original_default_text() const;

  // Returns pointer to the ptx for the requested compute capability.
  // Returns nullptr on failed lookup (if the requested version is not
  // available).
  // When the ptx is compressed, returns the decompressed ptx.
  const char *text(int compute_capability_major,
                   int compute_capability_minor) const;

  // Similar to text().
  // When the ptx is compressed, returns the original compressed ptx.
  const char *original_text(int compute_capability_major,
                            int compute_capability_minor) const;

  // Decompresses the PTX string using bzip2.
  static string DecompressPtx(const char *ptx);

 private:
  // PTX translation unit text contents in memory. The key is of as a tuple
  // "<cc_major>,<cc_minor>", i.e., "2,0", "3,0", "3,5". Because CC's
  // represented in this way have a clear sorting order, map::begin() will give
  // the lowest-numbered version available, i.e. the default.
  std::map<std::tuple<int, int>, const char *,
           bool (*)(const std::tuple<int, int> &, const std::tuple<int, int> &)>
      ptx_by_compute_capability_;

  // Stores all decompressed ptx strings, with original ptx string as keys.
  // It is marked as mutable for lazy decompression.
  mutable std::map<const char *, string> decompressed_ptx_;

  // Defines the minimum compute capability possible. Used when PTX has no
  // compute capability specified (in the single-PTX constructor).
  static const std::tuple<int, int> kMinimumCapability;

  DISALLOW_COPY_AND_ASSIGN(CudaPtxInMemory);
};

// Kernel loader specification for OpenCL text that resides on disk.
class OpenCLTextOnDisk : public OnDiskKernelLoaderSpec {
 public:
  OpenCLTextOnDisk(string filename, string kernelname);
  ~OpenCLTextOnDisk() override {}

  const char *CanonicalSuffix() const override { return ".ocl"; }

 private:
  DISALLOW_COPY_AND_ASSIGN(OpenCLTextOnDisk);
};

// Kernel loader specification for OpenCL binary that resides on disk.
class OpenCLBinaryOnDisk : public OnDiskKernelLoaderSpec {
 public:
  OpenCLBinaryOnDisk(string filename, string kernelname);
  ~OpenCLBinaryOnDisk() override {}

  const char *CanonicalSuffix() const override { return ".aocx"; }

 private:
  DISALLOW_COPY_AND_ASSIGN(OpenCLBinaryOnDisk);
};

// Kernel loader specification for OpenCL text that resides in memory.
class OpenCLTextInMemory : public KernelLoaderSpec {
 public:
  OpenCLTextInMemory(string text, string kernelname);
  ~OpenCLTextInMemory() override {}

  // Returns the OpenCL text contents.
  const string &text() const { return text_; }

 private:
  // OpenCL translation unit text contents in memory.
  string text_;

  DISALLOW_COPY_AND_ASSIGN(OpenCLTextInMemory);
};

// Kernel loader specification for a CUBIN blob that resides in memory.
class CudaCubinInMemory : public KernelLoaderSpec {
 public:
  CudaCubinInMemory(const char *bytes, string kernelname);
  ~CudaCubinInMemory() override {}

  const char *bytes() const { return bytes_; }

 private:
  const char *bytes_;

  DISALLOW_COPY_AND_ASSIGN(CudaCubinInMemory);
};

// Describes how to load a kernel on any subset of a number of target platforms.
class MultiKernelLoaderSpec {
 public:
  explicit MultiKernelLoaderSpec(size_t arity);

  // Returns the number of arguments that this kernel accepts.
  size_t arity() const { return arity_; }

  // Convenience getters for testing whether these platform variants have
  // kernel loader specifications available.
  bool has_cuda_ptx_on_disk() const { return cuda_ptx_on_disk_ != nullptr; }
  bool has_cuda_cubin_on_disk() const { return cuda_cubin_on_disk_ != nullptr; }
  bool has_cuda_cubin_in_memory() const {
    return cuda_cubin_in_memory_ != nullptr;
  }
  bool has_cuda_ptx_in_memory() const { return cuda_ptx_in_memory_ != nullptr; }
  bool has_ocl_text_on_disk() const { return ocl_text_on_disk_ != nullptr; }
  bool has_ocl_binary_on_disk() const { return ocl_binary_on_disk_ != nullptr; }
  bool has_ocl_text_in_memory() const { return ocl_text_in_memory_ != nullptr; }

  // Accessors for platform variant kernel load specifications.
  // Precondition: corresponding has_* is true.
  const CudaPtxOnDisk &cuda_ptx_on_disk() const {
    CHECK(has_cuda_ptx_on_disk());
    return *cuda_ptx_on_disk_;
  }
  const CudaCubinOnDisk &cuda_cubin_on_disk() const {
    CHECK(has_cuda_cubin_on_disk());
    return *cuda_cubin_on_disk_;
  }
  const CudaCubinInMemory &cuda_cubin_in_memory() const {
    CHECK(has_cuda_cubin_in_memory());
    return *cuda_cubin_in_memory_;
  }
  const CudaPtxInMemory &cuda_ptx_in_memory() const {
    CHECK(has_cuda_ptx_in_memory());
    return *cuda_ptx_in_memory_;
  }
  const OpenCLTextOnDisk &ocl_text_on_disk() const {
    CHECK(has_ocl_text_on_disk());
    return *ocl_text_on_disk_;
  }
  const OpenCLBinaryOnDisk &ocl_binary_on_disk() const {
    CHECK(has_ocl_binary_on_disk());
    return *ocl_binary_on_disk_;
  }
  const OpenCLTextInMemory &ocl_text_in_memory() const {
    CHECK(has_ocl_text_in_memory());
    return *ocl_text_in_memory_;
  }

  // Builder-pattern-like methods for use in initializing a
  // MultiKernelLoaderSpec. Each of these should be used at most once for a
  // single MultiKernelLoaderSpec object. See file comment for example usage.
  //
  // Note that the kernelname parameter must be consistent with the kernel in
  // the PTX or OpenCL being loaded. Also be aware that in CUDA C++ the kernel
  // name may be mangled by the compiler if it is not declared in an
  // extern "C" scope.
  MultiKernelLoaderSpec *AddOpenCLTextOnDisk(string filename,
                                             string kernelname);
  MultiKernelLoaderSpec *AddOpenCLBinaryOnDisk(string filename,
                                               string kernelname);
  MultiKernelLoaderSpec *AddOpenCLTextInMemory(string ocl_text,
                                               string kernelname);
  MultiKernelLoaderSpec *AddCudaPtxOnDisk(string filename,
                                          string kernelname);
  MultiKernelLoaderSpec *AddCudaCubinOnDisk(string filename,
                                            string kernelname);
  MultiKernelLoaderSpec *AddCudaCubinInMemory(const char *cubin_bytes,
                                              string kernelname);
  MultiKernelLoaderSpec *AddCudaPtxInMemory(string ptx,
                                            string kernelname);
  MultiKernelLoaderSpec *AddCudaCompressedPtxInMemory(
      string ptx, string kernelname);
  MultiKernelLoaderSpec *AddCudaPtxInMemory(
      std::initializer_list<CudaPtxInMemory::PtxSpec> spec_list,
      string kernelname);
  MultiKernelLoaderSpec *AddCudaCompressedPtxInMemory(
      std::initializer_list<CudaPtxInMemory::PtxSpec> spec_list,
      string kernelname);

 private:
  std::unique_ptr<CudaPtxOnDisk>
      cuda_ptx_on_disk_;  // PTX text that resides in a file.
  std::unique_ptr<CudaCubinOnDisk>
      cuda_cubin_on_disk_;  // Binary CUDA program in a file.
  std::unique_ptr<CudaCubinInMemory>
      cuda_cubin_in_memory_;  // Binary CUDA program in memory.
  std::unique_ptr<CudaPtxInMemory>
      cuda_ptx_in_memory_;  // PTX text that resides in memory.
  std::unique_ptr<OpenCLTextOnDisk>
      ocl_text_on_disk_;  // OpenCL text that resides on disk.
  std::unique_ptr<OpenCLBinaryOnDisk>
      ocl_binary_on_disk_;  // OpenCL binary that resides on disk.
  std::unique_ptr<OpenCLTextInMemory>
      ocl_text_in_memory_;  // OpenCL text that resides in memory.

  // Number of parameters that the kernel takes. (This is nicer to have in a
  // constexpr than having to determine it from the types via template
  // metaprogramming).
  size_t arity_;
};


#endif
