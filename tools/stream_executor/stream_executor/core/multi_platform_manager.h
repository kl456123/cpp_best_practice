#ifndef STREAM_EXECUTOR_CORE_MULTI_PLATFORM_MANAGER_H_
#define STREAM_EXECUTOR_CORE_MULTI_PLATFORM_MANAGER_H_
#include <string>

#include "stream_executor/utils/status.h"
#include "stream_executor/core/platform.h"

using std::string;

class MultiPlatformManager{
    public:
        // Registers a platform object, returns an error status if the platform is
        // already registered. The associated listener, if not null, will be used to
        // trace events for ALL executors for that platform.
        // Takes ownership of platform.
        static Status RegisterPlatform(std::unique_ptr<Platform> platform);
        // Retrieves the platform registered with the given platform name (e.g.
        // "CUDA", "OpenCL", ...) or id (an opaque, comparable value provided by the
        // Platform's Id() method).
        //
        // If the platform has not already been initialized, it will be initialized
        // with a default set of parameters.
        //
        // If the requested platform is not registered, an error status is returned.
        // Ownership of the platform is NOT transferred to the caller --
        // the MultiPlatformManager owns the platforms in a singleton-like fashion.
        static Status PlatformWithName(string target, Platform*);
        static Status PlatformWithId(const Platform::Id& id, Platform*);

        static Status InitializePlatformWithName(
                std::string target, const std::map<string, string>& options, Platform*);

        static Status InitializePlatformWithId(
                const Platform::Id& id, const std::map<string, string>& options,
                Platform* platform);
        // Interface for a listener that gets notfied at certain events.
        class Listener {
            public:
                virtual ~Listener() = default;
                // Callback that is invoked when a Platform is registered.
                virtual void PlatformRegistered(Platform* platform) = 0;
        };
        // Registers a listeners to receive notifications about certain events.
        // Precondition: No Platform has been registered yet.
        static Status RegisterListener(std::unique_ptr<Listener> listener);
};


#endif
