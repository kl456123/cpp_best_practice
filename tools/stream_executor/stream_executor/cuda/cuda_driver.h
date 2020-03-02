#ifndef STREAM_EXECUTOR_CUDA_CUDA_DRIVER_H_
#define STREAM_EXECUTOR_CUDA_CUDA_DRIVER_H_
#include <string>
#include <cstdint>
#include <cuda.h>

#include "stream_executor/utils/status.h"
#include "stream_executor/cuda/cuda_types.h"
#include "stream_executor/core/device_options.h"

typedef uint64_t uint64;
typedef int64_t int64;

namespace cuda{
    class CudaContext;
    class CudaDriver;

    class CudaDriver{
        public:
            // Wraps a call to cuInit with logging to help indicate what has gone wrong in
            // the case of failure. Safe to call multiple times; will be fast on all calls
            // after the first.
            // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__INITIALIZE.html#group__CUDA__INITIALIZE_1g0a2f1517e1bd8502c7194c3a8c134bc3
            static Status Init();

            // Returns the device associated with the given context.
            // device is an outparam owned by the caller, must not be null.
            // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g4e84b109eba36cdaaade167f34ae881e
            static Status DeviceFromContext(CudaContext* context, CudaDeviceHandle*);
            // Creates a new CUDA stream associated with the given context via
            // cuStreamCreate.
            // stream is an outparam owned by the caller, must not be null.
            // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1ga581f0c5833e21ded8b5a56594e243f4
            static bool CreateStream(CudaContext* context, CudaStreamHandle* stream);

            // Destroys a CUDA stream associated with the given context.
            // stream is owned by the caller, must not be null, and *stream is set to null
            // if the stream is successfully destroyed.
            // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g244c8833de4596bcd31a06cdf21ee758
            static void DestroyStream(CudaContext* context, CudaStreamHandle* stream);

            // Allocates a GPU memory space of size bytes associated with the given
            // context via cuMemAlloc.
            // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gb82d2a09844a58dd9e744dc31e8aa467
            static void* DeviceAllocate(CudaContext* context, uint64 bytes);
            // Deallocates a GPU memory space of size bytes associated with the given
            // context via cuMemFree.
            // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a
            static void DeviceDeallocate(CudaContext* context, void* location);
            // Allocates a unified memory space of size bytes associated with the given
            // context via cuMemAllocManaged.
            // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1gb347ded34dc326af404aa02af5388a32
            // (supported on CUDA only)
            static void* UnifiedMemoryAllocate(CudaContext* context, uint64 bytes);

            // Deallocates a unified memory space of size bytes associated with the given
            // context via cuMemFree.
            // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g89b3f154e17cc89b6eea277dbdf5c93a
            // (supported on CUDA only)
            static void UnifiedMemoryDeallocate(CudaContext* context, void* location);
            // Given a device ordinal, returns a device handle into the device outparam,
            // which must not be null.
            //
            // N.B. these device handles do not have a corresponding destroy function in
            // the CUDA driver API.
            static Status GetDevice(int device_ordinal, CudaDeviceHandle* device);
            // Given a device handle, returns the name reported by the driver for the
            // device.
            static Status GetDeviceName(CudaDeviceHandle device,
                    string* device_name);
            // Given a device to create a context for, returns a context handle into the
            // context outparam, which must not be null.
            //
            // N.B. CUDA contexts are weird. They are implicitly associated with the
            // calling thread. Current documentation on contexts and their influence on
            // userspace processes is given here:
            // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g65dc0012348bc84810e2103a40d8e2cf
            static Status CreateContext(int device_ordinal, CudaDeviceHandle device,
                    const DeviceOptions& device_options,
                    CudaContext** context);
            // Destroys the provided context via cuCtxDestroy.
            // Don't do this while clients could still be using the context, per the docs
            // bad things will happen.
            // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g27a365aebb0eb548166309f58a1e8b8e
            static void DestroyContext(CudaContext* context);

            // Launches a CUDA kernel via cuLaunchKernel.
            // TODO(leary) describe the structure of kernel_params and extra in a readable
            // way.
            // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html#group__CUDA__EXEC_1gb8f3dc3031b40da29d5f9a7139e52e15
            static Status LaunchKernel(
                    CudaContext* context, CudaFunctionHandle function, unsigned int grid_dim_x,
                    unsigned int grid_dim_y, unsigned int grid_dim_z,
                    unsigned int block_dim_x, unsigned int block_dim_y,
                    unsigned int block_dim_z, unsigned int shared_mem_bytes,
                    CudaStreamHandle stream, void** kernel_params, void** extra);

            // -- Synchronous memcopies.
            // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g4d32266788c440b0220b1a9ba5795169

            static Status SynchronousMemcpyD2H(CudaContext* context, void* host_dst,
                    CudaDevicePtr cuda_src, uint64 size);
            static Status SynchronousMemcpyH2D(CudaContext* context,
                    CudaDevicePtr cuda_dst,
                    const void* host_src, uint64 size);
            static Status SynchronousMemcpyD2D(CudaContext* context,
                    CudaDevicePtr cuda_dst,
                    CudaDevicePtr cuda_src, uint64 size);

            // -- Asynchronous memcopies.
            // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g56f30236c7c5247f8e061b59d3268362

            static bool AsynchronousMemcpyD2H(CudaContext* context, void* host_dst,
                    CudaDevicePtr cuda_src, uint64 size,
                    CudaStreamHandle stream);
            static bool AsynchronousMemcpyH2D(CudaContext* context, CudaDevicePtr cuda_dst,
                    const void* host_src, uint64 size,
                    CudaStreamHandle stream);
            static bool AsynchronousMemcpyD2D(CudaContext* context, CudaDevicePtr cuda_dst,
                    CudaDevicePtr cuda_src, uint64 size,
                    CudaStreamHandle stream);

            // -- Device-specific calls.

            // Returns the compute capability for the device; i.e (3, 5).
            // This is currently done via the deprecated device API.
            // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE__DEPRECATED.html#group__CUDA__DEVICE__DEPRECATED_1ge2091bbac7e1fb18c2821612115607ea
            // (supported on CUDA only)
            static Status GetComputeCapability(int* cc_major, int* cc_minor,
                    CudaDeviceHandle device);

            // Returns Gpu ISA version for the device; i.e 803, 900.
            // (supported on ROCm only)
            static Status GetCudaISAVersion(int* version, CudaDeviceHandle device);

            // Returns the number of multiprocessors on the device (note that the device
            // may be multi-GPU-per-board).
            static Status GetMultiprocessorCount(CudaDeviceHandle device, int* );

            // Returns the limit on number of threads that can be resident in a single
            // multiprocessor.
            static Status GetMaxThreadsPerMultiprocessor(
                    CudaDeviceHandle device, int64*);

            // Returns the limit on number of threads which may be resident for a single
            // block (cooperative thread array).
            static Status GetMaxThreadsPerBlock(CudaDeviceHandle device, int64* );

            // Returns the amount of shared memory available on a single GPU core (i.e.
            // SM on NVIDIA devices).
            static Status GetMaxSharedMemoryPerCore(
                    CudaDeviceHandle device, int64* );

            // Returns the amount of shared memory available for a single block
            // (cooperative thread array).
            static Status GetMaxSharedMemoryPerBlock(
                    CudaDeviceHandle device, int64*);

            // Returns the maximum supported number of registers per block.
            static Status GetMaxRegistersPerBlock(CudaDeviceHandle device, int64*);

            // Returns the number of threads per warp.
            static Status GetThreadsPerWarp(CudaDeviceHandle device, int64*);
            // Returns true if all stream tasks have completed at time of the call. Note
            // the potential for races around this call (if another thread adds work to
            // the stream immediately after this returns).
            static bool IsStreamIdle(CudaContext* context, CudaStreamHandle stream);

            // Returns whether ECC is enabled for the given GpuDeviceHandle via
            // cuDeviceGetattribute with CU_DEVICE_ATTRIBUTE_ECC_ENABLED.
            // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g9c3e1414f0ad901d3278a4d6645fc266
            static bool IsEccEnabled(CudaDeviceHandle device, bool* result);

            // Returns the total amount of memory available for allocation by the CUDA
            // context, in bytes, via cuDeviceTotalMem.
            static bool GetDeviceTotalMemory(CudaDeviceHandle device, uint64* result);

            // Returns the free amount of memory and total amount of memory, as reported
            // by cuMemGetInfo.
            // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0
            static bool GetDeviceMemoryInfo(CudaContext* context, int64* free,
                    int64* total);

            // Returns a PCI bus id string for the device.
            // [domain]:[bus]:[device].[function]
            // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g85295e7d9745ab8f0aa80dd1e172acfc
            static string GetPCIBusID(CudaDeviceHandle device);

            // -- Context- and device-independent calls.

            // Returns the number of visible CUDA device via cuDeviceGetCount.
            // This should correspond to the set of device ordinals available.
            // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html#group__CUDA__DEVICE_1g52b5ce05cb8c5fb6831b2c0ff2887c74
            static int GetDeviceCount();

            // Returns the driver version number via cuDriverGetVersion.
            // This is, surprisingly, NOT the actual driver version (e.g. 331.79) but,
            // instead, the CUDA toolkit release number that this driver is compatible
            // with; e.g. 6000 (for a CUDA 6.0 compatible driver) or 6050 (for a CUDA 6.5
            // compatible driver).
            //
            // http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VERSION.html#group__CUDA__VERSION_1g8b7a10395392e049006e61bcdc8ebe71
            static bool GetDriverVersion(int* driver_version);
            // Seam for injecting an error at CUDA initialization time for testing
            // purposes.
            static bool driver_inject_init_error_;
    };

    // CUDAContext wraps a cuda CUcontext handle, and includes a unique id. The
    // unique id is positive, and ids are not repeated within the process.
    class CudaContext {
        public:
            CudaContext(CUcontext context, int64 id) : context_(context), id_(id) {}

            CUcontext context() const { return context_; }
            int64 id() const { return id_; }

            // Disallow copying and moving.
            CudaContext(CudaContext&&) = delete;
            CudaContext(const CudaContext&) = delete;
            CudaContext& operator=(CudaContext&&) = delete;
            CudaContext& operator=(const CudaContext&) = delete;

        private:
            CUcontext const context_;
            const int64 id_;
    };
}



#endif
