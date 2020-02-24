#ifndef STREAM_EXECUTOR_PLATFORM_H_
#define STREAM_EXECUTOR_PLATFORM_H_
#include <string>

#include "stream_executor/utils/status.h"
#include "stream_executor/utils/macros.h"

class StreamExecutor;
class DeviceDescription;


using std::string;
struct StreamExecutorConfig{
    StreamExecutorConfig();
    // Simple ordinal-setting constructor.
    explicit StreamExecutorConfig(int ordinal);
    // The ordinal of the device to be managed by the returned StreamExecutor.
    int ordinal;
};

class Platform{
    public:
        virtual ~Platform();
        using Id = void*;

#define PLATFORM_DEFINE_ID(ID_VAR_NAME)                 \
        namespace{                                      \
            int plugin_id_value;                        \
        }                                               \
        const Platform::Id ID_VAR_NAME=&plugin_id_value;

        // Returns a key uniquely identifying this platform.
        virtual Id id() const = 0;

        // Name of this platform.
        virtual const string& Name() const = 0;

        // Returns the number of devices accessible on this platform.
        //
        // Note that, though these devices are visible, if there is only one userspace
        // context allowed for the device at a time and another process is using this
        // device, a call to ExecutorForDevice may return an error status.
        virtual int VisibleDeviceCount() const = 0;

        // Returns true iff the platform has been initialized.
        virtual bool Initialized() const;

        // Initializes the platform with a custom set of options. The platform must be
        // initialized before obtaining StreamExecutor objects.  The interpretation of
        // the platform_options argument is implementation specific.  This method may
        // return an error if unrecognized options are provided.  If using
        // MultiPlatformManager, this method will be called automatically by
        // InitializePlatformWithId/InitializePlatformWithName.
        virtual Status Initialize(
                const std::map<string, string>& platform_options);

        // Returns a populated DeviceDescription for the device at the given ordinal.
        // This should not require device initialization. Note that not all platforms
        // may support acquiring the DeviceDescription indirectly.
        //
        // Alternatively callers may call GetDeviceDescription() on the StreamExecutor
        // which returns a cached instance specific to the initialized StreamExecutor.
        virtual Status DescriptionForDevice(int ordinal, DeviceDescription*) const = 0;

        // Returns a device with the given ordinal on this platform with a default
        // plugin configuration or, if none can be found with the given ordinal or
        // there is an error in opening a context to communicate with the device, an
        // error status is returned.
        //
        // Ownership of the executor is NOT transferred to the caller --
        // the Platform owns the executors in a singleton-like fashion.
        virtual Status ExecutorForDevice(int ordinal, StreamExecutor*) = 0;
        // Returns a device or error, as above, with the specified plugins.
        //
        // Ownership of the executor is NOT transferred to the caller.
        // virtual Status ExecutorForDeviceWithPluginConfig(
        // int ordinal, const PluginConfig& plugin_config, StreamExecutor*) = 0;

        // Returns a device constructed with the options specified in "config".
        // Ownership of the executor is NOT transferred to the caller.
        virtual Status GetExecutor(
                const StreamExecutorConfig& config, StreamExecutor*) = 0;

        // Returns a device constructed with the options specified in "config" without
        // looking in or storing to the Platform's executor cache.
        // Ownership IS transferred to the caller.
        virtual Status GetUncachedExecutor(
                const StreamExecutorConfig& config, StreamExecutor*) = 0;
    protected:
        // SE_DISALLOW_COPY_AND_ASSIGN declares a constructor, which suppresses the
        // presence of the default constructor. This statement re-enables it, which
        // simplifies subclassing.
        Platform() = default;

    private:
        DISALLOW_COPY_AND_ASSIGN(Platform);
};
#endif
