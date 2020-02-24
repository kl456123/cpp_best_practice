#ifndef STREAM_EXECUTOR_CORE_STREAM_H_
#define STREAM_EXECUTOR_CORE_STREAM_H_
#include <cstdint>

#include "stream_executor/core/kernel.h"
#include "stream_executor/core/launch_dim.h"
#include "stream_executor/utils/status.h"
#include "stream_executor/utils/logging.h"

typedef uint64_t uint64;

// forward declaration
namespace internal{
    class StreamInterface;
}
class StreamExecutor;
class Timer;

class Stream{
    public:
        explicit Stream(StreamExecutor* parent);
        // Deallocates any stream resources that the parent StreamExecutor has
        // bestowed
        // upon this object.
        ~Stream();

        // Returns whether any errors have occurred while entraining work for this
        // stream.
        bool ok() const { return !InErrorState(); }

        // Initialize the stream. This must be performed before entraining any other
        // operations.
        Stream &Init();

        // Allocate temporary memories. The stream will deallocate them when blocked
        // or destroyed.
        // template <typename T>
        // Status AllocateTemporaryArray(uint64 element_count, TemporaryDeviceMemory*);

        // Entrains onto the stream of operations: a kernel launch with the given
        // (variadic) parameters for the invocation. These arguments can be things
        // like DeviceMemory or primitive types such as int. What arguments you may
        // pass to a given kernel are noted as the template parameters to the
        // TypedKernel type that the machocc compiler generates.
        //
        // Template parameters:
        //  Params...   The type list of formal parameters that the typed kernel
        //              expects, which is matched against Args...
        //  Args...     The deduced type list for passed actual arguments
        //
        // Implementation: A compile-time compatibility check is performed that has
        // some leniency versus an exact parameter pack match -- for example,
        // `const DeviceMemory<T>` is considered "pack compatible" with a
        // `const DeviceMemory<T>&` formal parameter; in part, because we don't have
        // perfect forwarding support without rvalue references. It also attempts to
        // spit out helpful static_assert error traces with information as to the
        // argument number and types that were mismatched.
        template <typename... Params, typename... Args>
            Stream &ThenLaunch(ThreadDim thread_dims, BlockDim block_dims,
                    const TypedKernel<Params...> &kernel, Args... args);
        // Record a "start" event for the interval timer at this point in the
        // stream's execution (relative to the previously and subsequently enqueued
        // items in the stream's execution). Streams may be started/stopped multiple
        // times.
        Stream &ThenStartTimer(Timer *t);

        // Record a "stop" event for the interval timer at this point in the
        // stream's execution. See also Stream::ThenStartTimer.
        Stream &ThenStopTimer(Timer *t);

        // (Synchronously) block the host code waiting for the operations
        // entrained on the stream (enqueued to this point in program
        // execution) to complete.
        //
        // Returns an OK status if the blocking was successful and the stream is ok().
        // Otherwise returns an error describing why the blocking failed.
        Status BlockHostUntilDone() ;

        // Returns the (opaque) platform-specific backing object. Ownership is not
        // transferred to the caller.
        internal::StreamInterface *implementation() { return implementation_.get(); }
        // Returns the StreamExecutor (parent object) associated with this stream.
        StreamExecutor *parent() const {
            CHECK(parent_ != nullptr);
            return parent_;
        }
        // Returns a debugging string "[stream=0x...,impl=0x...]".
        string DebugStreamPointers() const;
    private:
        // The StreamExecutor that supports the operation of this stream.
        StreamExecutor *parent_;
        // The platform-dependent implementation that the StreamExecutor interface
        // delegates to.
        std::unique_ptr<internal::StreamInterface> implementation_;
        // Whether Init() was successfully called to allocate this stream on the
        // underlying platform. It simply flips from 0 to 1 with a sanity check.
        // See StreamExecutor::AllocateStream.
        bool allocated_ ;

        // Whether all operations have entrained successfully to the current program
        // point.
        bool ok_ ;
        DISALLOW_COPY_AND_ASSIGN(Stream);

        bool InErrorState() const {
            return !ok_;
        }

        // Sets the error state if operation_retcode is false.
        // This is a useful shorthand for many stream routines.
        void CheckError(bool operation_retcode){
            if (operation_retcode) {
                return;
            }
            ok_ = false;
        }

        // Checks the status and logs the error message, if any.
        void CheckStatus(Status status);

        void SetError() { CheckError(false /* = operation_retcode */); }

        };



#endif
