#include <unordered_map>

#include "stream_executor/core/multi_platform_manager.h"
#include "stream_executor/utils/status_macros.h"
#include "stream_executor/utils/logging.h"
#include "stream_executor/utils/status.h"
#include "stream_executor/utils/errors.h"

namespace{
    class MultiPlatformManagerImpl{
        public:
            Status RegisterPlatform(std::unique_ptr<Platform> platform);

            Status PlatformWithName(std::string target, Platform**);

            Status PlatformWithId(const Platform::Id& id, Platform**);

            Status InitializePlatformWithName(
                    std::string target, const std::map<string, string>& options, Platform**);

            Status InitializePlatformWithId(
                    const Platform::Id& id, const std::map<string, string>& options, Platform**);

            using Listener = MultiPlatformManager::Listener;
            Status RegisterListener(std::unique_ptr<Listener> listener);

        private:
            // Looks up the platform object with the given name.  Assumes the Platforms
            // mutex is held.
            Status LookupByNameLocked(std::string target, Platform**);

            // Looks up the platform object with the given id.  Assumes the Platforms
            // mutex is held.
            Status LookupByIdLocked(const Platform::Id& id, Platform**);

            std::vector<std::unique_ptr<Listener>> listeners_ ;
            std::unordered_map<Platform::Id, Platform*> id_map_ ;
            std::unordered_map<string, Platform*> name_map_ ;
    };

    // implementation
    Status MultiPlatformManagerImpl::RegisterPlatform(
            std::unique_ptr<Platform> platform) {
        std::string key = platform->Name();
        if(name_map_.find(key)!=name_map_.end()){
            return errors::Internal(
                    "platform is already registered with name: \""+platform->Name()+"\"");
        }
        Platform* platform_ptr=nullptr;
        CHECK(id_map_.emplace(platform->id(), platform_ptr).second);

        name_map_[key] = platform.release();
        for(const auto& listener: listeners_){
            listener->PlatformRegistered(platform_ptr);
        }
        return Status::OK();
    }

    Status MultiPlatformManagerImpl::PlatformWithName(
            std::string target, Platform** platform) {
        // SE_ASSIGN_OR_RETURN(platform, LookupByNameLocked(target));
        SE_RETURN_IF_ERROR(LookupByNameLocked(target, platform));
        if (!(*platform)->Initialized()) {
            SE_RETURN_IF_ERROR((*platform)->Initialize({}));
        }

        return Status::OK();
    }

    Status MultiPlatformManagerImpl::PlatformWithId(
            const Platform::Id& id, Platform** platform) {

        // SE_ASSIGN_OR_RETURN(platform, LookupByIdLocked(id));
        SE_RETURN_IF_ERROR(LookupByIdLocked(id, platform));
        if (!(*platform)->Initialized()) {
            SE_RETURN_IF_ERROR((*platform)->Initialize({}));
        }

        return Status::OK();
    }

    Status MultiPlatformManagerImpl::InitializePlatformWithName(
            std::string target, const std::map<string, string>& options,
            Platform** platform) {

        // SE_ASSIGN_OR_RETURN(platform, LookupByNameLocked(target));
        SE_RETURN_IF_ERROR(LookupByNameLocked(target, platform));
        if ((*platform)->Initialized()) {
            return errors::FailedPrecondition(
                    "platform \""+ target+ "\" is already initialized");
        }

        SE_RETURN_IF_ERROR((*platform)->Initialize(options));

        return Status::OK();
    }

    Status MultiPlatformManagerImpl::InitializePlatformWithId(
            const Platform::Id& id, const std::map<string, string>& options,
            Platform** platform) {
        // SE_ASSIGN_OR_RETURN(platform, LookupByIdLocked(id));
        SE_RETURN_IF_ERROR(LookupByIdLocked(id, platform));
        if ((*platform)->Initialized()) {
            return errors::FailedPrecondition(
                    std::string("platform with id ") + " is already initialized");
        }

        SE_RETURN_IF_ERROR((*platform)->Initialize(options));

        return Status::OK();
    }
    Status MultiPlatformManagerImpl::RegisterListener(
            std::unique_ptr<Listener> listener) {
        CHECK(id_map_.empty());
        CHECK(name_map_.empty());
        listeners_.push_back(std::move(listener));
        return Status::OK();
    }
    Status MultiPlatformManagerImpl::LookupByNameLocked(
            std::string target,Platform** platform) {
        auto it = name_map_.find(target);
        if (it == name_map_.end()) {
            return errors::NotFound(
                    "Could not find registered platform with name: \""+ target+
                    "\"");
        }
        *platform = it->second;
        return Status::OK();
    }

    Status MultiPlatformManagerImpl::LookupByIdLocked(
            const Platform::Id& id, Platform** platform) {
        auto it = id_map_.find(id);
        if (it == id_map_.end()) {
            return errors::NotFound(
                    "could not find registered platform with id: ");
        }
        *platform = it->second;
        return Status::OK();
    }



    MultiPlatformManagerImpl& Impl() {
        static MultiPlatformManagerImpl* impl = new MultiPlatformManagerImpl;
        return *impl;
    }
}


/*static*/ Status MultiPlatformManager::RegisterPlatform(
        std::unique_ptr<Platform> platform) {
    return Impl().RegisterPlatform(std::move(platform));
}

/*static*/ Status MultiPlatformManager::PlatformWithName(
        std::string target, Platform** platform) {
    SE_RETURN_IF_ERROR(Impl().PlatformWithName(target, platform));
    return Status::OK();
}

/*static*/ Status MultiPlatformManager::PlatformWithId(
        const Platform::Id& id, Platform** platform) {
    SE_RETURN_IF_ERROR(Impl().PlatformWithId(id, platform));
    return Status::OK();
}

/*static*/ Status MultiPlatformManager::InitializePlatformWithName(
        std::string target, const std::map<string, string>& options,
        Platform** platform) {
    SE_RETURN_IF_ERROR(Impl().InitializePlatformWithName(target, options, platform));
    return Status::OK();
}

/*static*/ Status MultiPlatformManager::InitializePlatformWithId(
        const Platform::Id& id, const std::map<string, string>& options,
        Platform** platform) {
    SE_RETURN_IF_ERROR(Impl().InitializePlatformWithId(id, options, platform));
    return Status::OK();
}

/*static*/ Status MultiPlatformManager::RegisterListener(
        std::unique_ptr<Listener> listener) {
    return Impl().RegisterListener(std::move(listener));
}
