#include "session/utils/random.h"
#include <random>

namespace random_utils{

    namespace {
        std::mt19937_64* InitRngWithRandomSeed() {
            std::random_device device("/dev/urandom");
            return new std::mt19937_64(device());
        }
        std::mt19937_64 InitRngWithDefaultSeed() { return std::mt19937_64(); }

    }  // anonymous namespace

    uint64_t New64() {
        static std::mt19937_64* rng = InitRngWithRandomSeed();
        return (*rng)();
    }

    uint64_t New64DefaultSeed() {
        static std::mt19937_64 rng = InitRngWithDefaultSeed();
        return rng();
    }

}  // namespace random
