#ifndef STREAM_EXECUTOR_CORE_KERNEL_H_
#define STREAM_EXECUTOR_CORE_KERNEL_H_
#include <memory>
#include <string>
#include <vector>

#include "stream_executor/utils/macros.h"
#include "stream_executor/core/device_memory.h"

using std::string;

namespace internal{
    class KernelInterface;
}

class StreamExecutor;


class KernelMetadata{
    public:
};


class KernelBase{
    public:
        KernelBase(KernelBase &&from);

        explicit KernelBase(StreamExecutor *parent);

        KernelBase(StreamExecutor *parent, internal::KernelInterface *implementation);

        ~KernelBase();

        // Returns the number of parameters that this kernel accepts. (Arity refers to
        // nullary, unary, ...).
        unsigned Arity() const;

        // Returns the StreamExecutor that represents the platform this kernel
        // executes upon.
        StreamExecutor *parent() const { return parent_; }
        // Returns a non-const pointer to the (opaque) platform-dependent
        // implementation.
        internal::KernelInterface *implementation() { return implementation_.get(); }
        void set_metadata(const KernelMetadata &metadata) { metadata_ = metadata; }
        const KernelMetadata &metadata() const { return metadata_; }
        void set_name(std::string name);
        const string &name() const { return name_; }
        const string &demangled_name() const { return demangled_name_; }
    private:
        // The StreamExecutor that loads this kernel object.
        StreamExecutor *parent_;

        // Implementation delegated to for platform-specific functionality.
        std::unique_ptr<internal::KernelInterface> implementation_;

        string name_;
        string demangled_name_;

        KernelMetadata metadata_;

        DISALLOW_COPY_AND_ASSIGN(KernelBase);
};

// Whether T is a DeviceMemory-family pointer.
template <typename T>
struct IsDeviceMemoryPointer {
    static constexpr bool value = false;
};

template <typename U>
struct IsDeviceMemoryPointer<DeviceMemory<U> *> {
    static constexpr bool value = true;
};

template <>
struct IsDeviceMemoryPointer<DeviceMemoryBase *> {
    static constexpr bool value = true;
};

// Whether T is a DeviceMemory-family value-like thing (which includes a
// reference). This trait is useful because we pack values in the same manner as
// references.
template <typename T>
struct IsDeviceMemoryValueLike {
    static constexpr bool value = false;
};

template <typename U>
struct IsDeviceMemoryValueLike<DeviceMemory<U> &> {
    static constexpr bool value = true;
};

// We need to treat SharedDeviceMemory types differently than other DeviceMemory
// types (since they maintain no allocations), hence these specializations.
template <typename U>
struct IsDeviceMemoryValueLike<SharedDeviceMemory<U> &> {
    static constexpr bool value = false;
};

template <>
struct IsDeviceMemoryValueLike<DeviceMemoryBase &> {
    static constexpr bool value = true;
};

template <typename U>
struct IsDeviceMemoryValueLike<DeviceMemory<U>> {
    static constexpr bool value = true;
};

template <typename U>
struct IsDeviceMemoryValueLike<SharedDeviceMemory<U>> {
    static constexpr bool value = false;
};

template <>
struct IsDeviceMemoryValueLike<DeviceMemoryBase> {
    static constexpr bool value = true;
};

template <typename U>
struct IsSharedDeviceMemory {
    static constexpr bool value = false;
};

template <typename U>
struct IsSharedDeviceMemory<SharedDeviceMemory<U> &> {
    static constexpr bool value = true;
};

template <typename U>
struct IsSharedDeviceMemory<SharedDeviceMemory<U>> {
    static constexpr bool value = true;
};

// Basic data about a kernel argument.
struct KernelArg {
    bool is_shared;
    const void *address;
    size_t size;
};


class KernelArgIterator {
    public:
        KernelArgIterator(int number_of_argument_addresses,
                int number_of_shared_memory_arguments,
                const void *const *arg_addresses_data,
                const size_t *arg_sizes_data,
                const size_t *shmem_bytes_data,
                const size_t *shmem_indices_data)
            : arg_index_(0),
            number_of_arguments_(number_of_argument_addresses +
                    number_of_shared_memory_arguments),
            arg_address_iter_(arg_addresses_data),
            arg_size_iter_(arg_sizes_data),
            shmem_bytes_iter_(shmem_bytes_data),
            shmem_indices_iter_(shmem_indices_data),
            shmem_indices_end_(shmem_indices_data +
                    number_of_shared_memory_arguments) {}

        // Returns true if another argument is present in the iterator.
        bool has_next() { return arg_index_ < number_of_arguments_; }

        // Returns the next argument in the iterator.
        //
        // Returns a default-constructed KernelArg if there is no next argument.
        KernelArg next() {
            KernelArg result = {};
            if (!has_next()) {
                return result;
            } else if ((shmem_indices_iter_ != shmem_indices_end_) &&
                    (arg_index_ == *shmem_indices_iter_)) {
                result.is_shared = true;
                result.address = nullptr;
                result.size = *shmem_bytes_iter_;
                ++shmem_indices_iter_;
                ++shmem_bytes_iter_;
            } else {
                result.is_shared = false;
                result.address = *arg_address_iter_;
                result.size = *arg_size_iter_;
                ++arg_address_iter_;
                ++arg_size_iter_;
            }
            ++arg_index_;
            return result;
        }

    private:
        size_t arg_index_;
        size_t number_of_arguments_;
        const void *const *arg_address_iter_;
        const size_t *arg_size_iter_;
        const size_t *shmem_bytes_iter_;
        const size_t *shmem_indices_iter_;
        const size_t *const shmem_indices_end_;
};
// Base class for KernelArgsArray.
//
// Supports all the getter methods that do not depend on the compile-time number
// of arguments template parameter.
//
// This class exists as a way to pass kernel arguments to
// StreamExecutorInterface::Launch. That Launch method is virtual, so it can't
// be templated to accept any KernelArgsArray type, therefore a reference to
// this base type is passed instead.
//
// Performance is not a concern here because each of these methods will be
// called at most once per kernel launch. Past performance concerns with
// KernelArgsArray have been in reference to the argument packing routines which
// are called once per kernel argument. Those packing routines are now handled
// by the templated KernelArgsArray subclass of this class where they can take
// advantage of compile-time knowledge of the number of arguments in order to be
// very efficient.
class KernelArgsArrayBase {
    public:
        virtual ~KernelArgsArrayBase() = default;

        // Gets the number of arguments added so far, including shared memory
        // arguments.
        virtual size_t number_of_arguments() const = 0;

        // Gets the total number of shared memory bytes added so far.
        virtual uint64_t number_of_shared_bytes() const = 0;

        // Gets the list of argument addresses.
        virtual std::vector<const void *> argument_addresses() const = 0;

        // Gets an iterator to the arguments in the array.
        virtual KernelArgIterator arg_iterator() const = 0;
};

// A list of arguments for a kernel call.
//
// The template parameter kNumArgs is the maximum number of arguments which can
// be stored in the list.
//
// Contains a list of addresses for non-shared-memory arguments and a list of
// sizes for shared-memory arguments. Since the shared-memory arguments may be
// interspersed with the non-shared-memory arguments, it also stores a list of
// the indices at which the shared-memory arguments appeared.
//
// For example, if the argument address list contains {a, b, c, d, e}, the
// shared-memory arguments list contains the sizes of {A, B, C}, and the
// shared-memory indices list contains {0, 3, 5}, then the original list of
// arguments was {A, a, b, B, c, C, d, e}.
//
// This way of storing the arguments makes CUDA kernel calls efficient because
// they only require the argument address list and the total number of shared
// bytes, but it also makes it possible for OpenCL kernel calls because they
// depend on the location of each shared-memory argument and its size.
//
// Note that the code for adding arguments has been identified as a performance
// hotspot in some real-world applications so this structure has been optimized
// for the performance of argument adding.
template <size_t kNumArgs>
class KernelArgsArray : public KernelArgsArrayBase {
    public:
        explicit KernelArgsArray()
            : total_shared_memory_bytes_(0),
            number_of_argument_addresses_(0),
            number_of_shared_memory_arguments_(0) {}

        // Adds an argument to the list.
        //
        // Note that the address of the argument is stored, so the input must not go
        // out of scope before the instance of this class that calls this method does.
        template <typename T>
            void add_argument(const T &arg) {
                argument_addresses_[number_of_argument_addresses_] =
                    static_cast<const void *>(&arg);
                argument_sizes_[number_of_argument_addresses_] = sizeof(arg);
                ++number_of_argument_addresses_;
            }

        // Adds a device memory argument to the list.
        void add_device_memory_argument(const DeviceMemoryBase &arg) {
            const void **copy_ptr =
                &device_memory_opaque_pointers_[number_of_argument_addresses_];
            *copy_ptr = arg.opaque();
            argument_addresses_[number_of_argument_addresses_] = copy_ptr;
            argument_sizes_[number_of_argument_addresses_] = sizeof(void *);
            ++number_of_argument_addresses_;
        }

        // Adds a shared memory argument to the list.
        //
        // The only significant information about a shared argument is its size, so
        // that is the only parameter in this function.
        void add_shared_bytes(size_t number_of_bytes) {
            shared_memory_indices_[number_of_shared_memory_arguments_] =
                number_of_argument_addresses_ + number_of_shared_memory_arguments_;
            shared_memory_bytes_[number_of_shared_memory_arguments_] = number_of_bytes;
            ++number_of_shared_memory_arguments_;
            total_shared_memory_bytes_ += number_of_bytes;
        }

        // Gets the number of arguments added so far, including shared memory
        // arguments.
        size_t number_of_arguments() const override {
            return number_of_argument_addresses_ + number_of_shared_memory_arguments_;
        }

        // Gets the total number of shared memory bytes added so far.
        uint64_t number_of_shared_bytes() const override {
            return total_shared_memory_bytes_;
        }

        // Gets the list of argument addresses.
        std::vector<const void *> argument_addresses() const override {
            // return std::vector<const void *>({argument_addresses_.data(),
                    // number_of_argument_addresses_});
            return {};
        }

        // Gets an iterator to the arguments in the array.
        KernelArgIterator arg_iterator() const override {
            return KernelArgIterator(
                    number_of_argument_addresses_, number_of_shared_memory_arguments_,
                    argument_addresses_.data(), argument_sizes_.data(),
                    shared_memory_bytes_.data(), shared_memory_indices_.data());
        }

    private:
        // A place to store copies of opaque pointers from device memory arguments.
        std::array<const void *, kNumArgs> device_memory_opaque_pointers_;

        // Addresses for non-shared-memory arguments.
        std::array<const void *, kNumArgs> argument_addresses_;

        // Sizes for non-shared-memory arguments.
        std::array<size_t, kNumArgs> argument_sizes_;

        // Size in bytes for each shared memory argument.
        std::array<size_t, kNumArgs> shared_memory_bytes_;

        // Indices in the arguments array for shared memory arguments.
        std::array<size_t, kNumArgs> shared_memory_indices_;

        // Total of all shared memory sizes.
        size_t total_shared_memory_bytes_;

        // Number of significant entries in argument_addresses_ and argument_sizes_.
        size_t number_of_argument_addresses_;

        // Number of significant entries in shared_memory_bytes_ and
        // shared_memory_indices_.
        size_t number_of_shared_memory_arguments_;
};

// Typed variant of KernelBase, like a typed device function pointer. See the
// file comment for details and example usage.
//
// This class contains template metaprogramming magic to type check the
// parameters passed to a kernel launch are acceptable, and subsequently pack
// them into a form which can be used by the StreamExecutorInterface
// implementation. (i.e.  CUDA and OpenCL both bind void*s with associated
// sizes as kernel arguments.)
//
// Thread-compatible.
template <typename... Params>
class TypedKernel : public KernelBase {
    public:
        static constexpr size_t kNumberOfParameters = sizeof...(Params);
        // Delegates to KernelBase::KernelBase(), see that constructor.
        explicit TypedKernel(StreamExecutor *parent) : KernelBase(parent) {}

    private:
        // Stream needs access to the specific parameter-packing functionality that
        // the TypedKernel provides for its corresponding type signature (and no other
        // type signatures).
        friend class Stream;

        // This is the main entry point into the magic. Packs the parameters (which
        // must type check against the class template) into the args and sizes
        // arrays.
        //
        // Const refs are taken as parameters on all of the handlers to avoid
        // implicit type promotion of integers.
        //
        // WARNING: as a performance optimization this method may store pointers to
        // some of the input parameters in the kernel args structure, so any params
        // passed into this method must live at least as long as the kernel args
        // structure.
        void PackParams(KernelArgsArray<kNumberOfParameters> *args,
                Params &... params) const {
            PackOneParamFromList(args, params...);
        }

        template <typename T, typename... RestOfParams>
            void PackOneParamFromList(KernelArgsArray<kNumberOfParameters> *args,
                    const T &arg, const RestOfParams &... rest) const {
                PackOneParam(args, arg);
                PackOneParamFromList(args, rest...);
            }

        // Base case for variadic template expansion - nothing to do!
        void PackOneParamFromList(KernelArgsArray<kNumberOfParameters> *args) const {}

        // Packs one (non-DeviceMemoryBase) parameter into the arg and sizes array.
        // The enable_if<> is for excluding DeviceMemoryBase args, which have a
        // separate implementation below.
        template <typename T>
            void PackOneParam(
                    KernelArgsArray<kNumberOfParameters> *args, const T &arg,
                    typename std::enable_if<!IsDeviceMemoryValueLike<T>::value &&
                    !IsDeviceMemoryPointer<T>::value &&
                    !IsSharedDeviceMemory<T>::value>::type * =
                    nullptr) const {
                static_assert(!std::is_pointer<T>::value,
                        "cannot pass raw pointer to the device");
                static_assert(!std::is_convertible<T, DeviceMemoryBase>::value,
                        "cannot pass device memory as a normal value");
                args->add_argument(arg);
            }

        // DeviceMemoryBase family reference override.
        template <typename T>
            void PackOneParam(
                    KernelArgsArray<kNumberOfParameters> *args, const T &arg,
                    typename std::enable_if<IsDeviceMemoryValueLike<T>::value>::type * =
                    nullptr) const {
                args->add_device_memory_argument(arg);
            }

        // DeviceMemoryBase family pointer override.
        template <typename T>
            void PackOneParam(
                    KernelArgsArray<kNumberOfParameters> *args, T arg,
                    typename std::enable_if<IsDeviceMemoryPointer<T>::value>::type * =
                    nullptr) const {
                DeviceMemoryBase *ptr = static_cast<DeviceMemoryBase *>(arg);
                args->add_device_memory_argument(*ptr);
            }

        // Dynamic shared device memory has a size, but no associated allocation on
        // the host; internally, the device will allocate storage.
        template <typename T>
            void PackOneParam(
                    KernelArgsArray<kNumberOfParameters> *args, T arg,
                    typename std::enable_if<IsSharedDeviceMemory<T>::value>::type * =
                    nullptr) const {
                args->add_shared_bytes(arg.size());
            }

        DISALLOW_COPY_AND_ASSIGN(TypedKernel);
};

// Template metaprogramming helper type that helps us produce better error
// messages at compile time when the are mismatches between the parameter
// type list and the argument type list.
template <typename ParamTuple, typename ArgTuple>
struct KernelInvocationChecker {
    // Whether the parameter tuple and argument tuple match in length.
    static constexpr bool kLengthMatches =
        std::tuple_size<ParamTuple>::value == std::tuple_size<ArgTuple>::value;

    // The (matching) length of the parameters and arguments type lists.
    static constexpr int kTupleLength =
        static_cast<int>(std::tuple_size<ArgTuple>::value);

    // Helper trait to say whether the parameter wants a DeviceMemory-reference
    // compatible type. This is for inexact type matches, so that it doesn't have
    // to be precisely a const DeviceMemory<T>&, but can also be a value that
    // represents the same.
    template <typename ParamType, typename ArgType>
        struct IsCompatibleDeviceMemoryRef {
            static constexpr bool value = false;
        };

    // See type trait definition above.
    template <typename U>
        struct IsCompatibleDeviceMemoryRef<const DeviceMemory<U> &, DeviceMemory<U>> {
            static constexpr bool value = true;
        };

    // See type trait definition above.
    template <typename U>
        struct IsCompatibleDeviceMemoryRef<const SharedDeviceMemory<U> &,
        SharedDeviceMemory<U>> {
            static constexpr bool value = true;
        };

    // Returns whether ParamT and ArgT are compatible for data parallel kernel
    // parameter packing without any assert functionality.
    template <typename ParamT, typename ArgT>
        static constexpr bool CompatibleNoAssert() {
            return std::is_same<typename std::remove_const<ParamT>::type,
                   ArgT>::value ||
                       IsCompatibleDeviceMemoryRef<ParamT, ArgT>::value;
        }

    // Checks whether ParamT and ArgT are compatible for data parallel kernel
    // parameter packing. kArgumentNumber is unused, it just for error display.
    //
    // NOTE: if you encounter an error here, you can see the mismatch by looking
    // at the end of the last error message, which will be of the form:
    //
    //    ...::Compatible<const stream_executor::DeviceMemory<OneThing> &,
    //                    stream_executor::DeviceMemory<AnotherThing>, true,
    //                    0>'
    //    requested here
    //
    // This means that the 0th argument you passed to the kernel invocation should
    // have been DeviceMemory<OneThing> but was observed to be
    // DeviceMemory<AnotherThing>.
    template <typename ParamT, typename ArgT, bool kShouldStaticAssert,
             int kArgumentNumber>
                 static constexpr bool Compatible() {
                     static_assert(
                             kShouldStaticAssert ? CompatibleNoAssert<ParamT, ArgT>() : true,
                             "parameter type (LHS) is not compatible with argument type (RHS)");
                     return CompatibleNoAssert<ParamT, ArgT>();
                 }

    // Checks the parameter/argument match at kArgumentNumber for an out of bounds
    // argument number.
    //
    // This is the base case: we've run out of argument to check, so we're all
    // good.
    template <int kArgumentNumber, bool kShouldStaticAssert>
        static constexpr bool CheckParam(
                typename std::enable_if<(kArgumentNumber < 0)>::type *dummy = nullptr) {
            return true;
        }

    // Checks the parameter/argument match at kArgumentNumber.
    // kShouldStaticAssert determines whether to assert out on a mismatch, or just
    // yield the constexpr boolean value.
    template <int kArgumentNumber, bool kShouldStaticAssert>
        static constexpr bool CheckParam(
                typename std::enable_if<kArgumentNumber >= 0>::type *dummy = nullptr) {
            typedef typename std::tuple_element<kArgumentNumber, ParamTuple>::type
                ParamT;
            typedef typename std::tuple_element<kArgumentNumber, ArgTuple>::type ArgT;
            return Compatible<ParamT, ArgT, kShouldStaticAssert, kArgumentNumber>() &&
                CheckParam<kArgumentNumber - 1, kShouldStaticAssert>();
        }

    // Checks the parameters/arguments for match, but doesn't static assert out.
    // This is useful for testing/inspecting whether a set of parameters match in
    // things like tests.
    static constexpr bool CheckAllNoStaticAssert() {
        return kLengthMatches && CheckParam<kTupleLength - 1, false>();
    }

    // Checks the parameters and static asserts out with a helpful error message
    // (and useful template parameters in the instantiation stack) if there is an
    // error.
    static constexpr bool CheckAllStaticAssert() {
        static_assert(kLengthMatches,
                "argument length mismatched against typed kernel parameters");
        return kLengthMatches && CheckParam<kTupleLength - 1, true>();
    }
};

// This is a convenience type for checking whether a typed kernel matches
// against a type list.
template <typename KernelT, typename... Params>
struct KernelParamsOk {
    static constexpr bool kResult = false;
};

// See above.
template <typename... Params, typename... Args>
struct KernelParamsOk<TypedKernel<Params...>, Args...> {
    static constexpr bool kResult = KernelInvocationChecker<
        std::tuple<Params...>, std::tuple<Args...>>::CheckAllNoStaticAssert();
};



#endif
