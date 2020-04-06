#define EIGEN_USE_SYCL
#define EIGEN_USE_THREADS
// #define EIGEN_USE_GPU

#define GL_GLEXT_PROTOTYPES

#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <eigen3/unsupported/Eigen/CXX11/ThreadPool>

#include <eigen3/unsupported/Eigen/OpenGLSupport>
#include <iostream>

using std::string;
// using namespace cl;

class SYCLAllocator{
    public:
        SYCLAllocator(Eigen::QueueInterface* queue);
        ~SYCLAllocator();
        void* AllocateRaw(size_t alignment, size_t num_bytes);
        void DeallocateRaw(void* ptr);
        void Synchronize() {
            if (sycl_device_) {
                sycl_device_->synchronize();
            }
        }

        // Clear the SYCL device used by the Allocator
        void ClearSYCLDevice() {
            if (sycl_device_) {
                delete sycl_device_;
                sycl_device_ = nullptr;
            }
        }

        bool Ok()const {return sycl_device_ && sycl_device_->ok();}
        Eigen::SyclDevice* getSyclDevice() { return sycl_device_; }
    private:
        Eigen::SyclDevice* sycl_device_;  // owned
};


SYCLAllocator::SYCLAllocator(Eigen::QueueInterface* queue)
    :sycl_device_(new Eigen::SyclDevice(queue)){
        cl::sycl::queue& sycl_queue = sycl_device_->sycl_queue();
        const cl::sycl::device& device = sycl_queue.get_device();
    }

SYCLAllocator::~SYCLAllocator(){
    if(sycl_device_){
        delete sycl_device_;
    }
}

void* SYCLAllocator::AllocateRaw(size_t alignment, size_t num_bytes){
    assert(sycl_device_);
    if (num_bytes == 0) {
        // Cannot allocate no bytes in SYCL, so instead allocate a single byte
        num_bytes = 1;
    }
    auto p = sycl_device_->allocate(num_bytes);
    const auto& allocated_buffer = sycl_device_->get_sycl_buffer(p);
    const std::size_t bytes_allocated = allocated_buffer.get_range().size();
    return p;
}

void SYCLAllocator::DeallocateRaw(void* ptr) {
    if (sycl_device_) {
        const auto& buffer_to_delete = sycl_device_->get_sycl_buffer(ptr);
        const std::size_t dealloc_size = buffer_to_delete.get_range().size();
        sycl_device_->deallocate(ptr);
    }
}

class GSYCLInterface{
    std::vector<Eigen::QueueInterface*> m_queue_interface_;  // owned
    std::vector<SYCLAllocator*> m_sycl_allocator_;           // owned
    GSYCLInterface() {
        bool found_device = false;
        auto device_list = Eigen::get_sycl_supported_devices();
        // Obtain list of supported devices from Eigen
        for (const auto& device : device_list) {
            if (device.is_gpu()) {
                // returns first found GPU
                AddDevice(device);
                found_device = true;
            }
        }

        if (!found_device) {
            // Currently Intel GPU is not supported
            std::cout << "No OpenCL GPU found that is supported by "
                << "ComputeCpp/triSYCL, trying OpenCL CPU";
        }
    }

    ~GSYCLInterface() {
        for (auto p : m_sycl_allocator_) {
            p->Synchronize();
            p->ClearSYCLDevice();
            // Cannot delete the Allocator instances, as the Allocator lifetime
            // needs to exceed any Tensor created by it. There is no way of
            // knowing when all Tensors have been deallocated, as they are
            // RefCounted and wait until all instances of a Tensor have been
            // destroyed before calling Allocator.Deallocate. This could happen at
            // program exit, which can set up a race condition between destroying
            // Tensors and Allocators when the program is cleaning up.
        }
        m_sycl_allocator_.clear();
        for (auto p : m_queue_interface_) {
            p->deallocate_all();
            delete p;
        }
        m_queue_interface_.clear();
    }

    void AddDevice(const cl::sycl::device& d) {
        m_queue_interface_.push_back(new Eigen::QueueInterface(d));
        m_sycl_allocator_.push_back(new SYCLAllocator(m_queue_interface_.back()));
    }
    public:
    static const GSYCLInterface* instance() {
        // c++11 guarantees that this will be constructed in a thread safe way
        static const GSYCLInterface instance;
        return &instance;
    }

    SYCLAllocator* GetSYCLAllocator(size_t i = 0) const {
        if (!m_sycl_allocator_.empty()) {
            return m_sycl_allocator_[i];
        } else {
            std::cerr << "No cl::sycl::device has been added" << std::endl;
            return nullptr;
        }
    }

    Eigen::QueueInterface* GetQueueInterface(size_t i = 0) const {
        if (!m_queue_interface_.empty()) {
            return m_queue_interface_[i];
        } else {
            std::cerr << "No cl::sycl::device has been added" << std::endl;
            return nullptr;
        }
    }


};

void test_allocator(){
    // sycl
    auto syclInterface = GSYCLInterface::instance();
    // allocator
    auto syclAllocator  = syclInterface->GetSYCLAllocator();
    void* p = syclAllocator->AllocateRaw(1<<5, 10*sizeof(float));
    float* typed_p = reinterpret_cast<float*>(p);

    Eigen::TensorMap<Eigen::Tensor<float, 1>> t4(typed_p, 10);

    void* p2 = syclAllocator->AllocateRaw(1<<5, 10*sizeof(float));
    float* typed_p2 = reinterpret_cast<float*>(p2);

    Eigen::TensorMap<Eigen::Tensor<float, 1>> t5(typed_p2, 10);

    t4.device(*(syclAllocator->getSyclDevice())) = t4.constant(102.f)+t5.constant(201.f);
}

void test_tensor(){
    Eigen::Tensor<float, 3> t1(2,3,4);
    Eigen::Tensor<float, 3> t2(2,3,4);
    Eigen::Tensor<float, 3> t3(2,3,4);
    // Eigen::DefaultDevice d;
    // Create the Eigen ThreadPoolDevice.

    // Eigen::ThreadPool pool(4);
    // Eigen::ThreadPoolDevice d(&pool, 4 [> number of threads to use <]);

    // sycl
    auto syclInterface = GSYCLInterface::instance();
    // allocator
    auto syclAllocator  = syclInterface->GetSYCLAllocator();
    Eigen::SyclDevice& d = *syclAllocator->getSyclDevice();

    // Eigen::GpuDevice device();
    t3.device(d)= t1+t2+1.f;
    std::cout<<t3<<std::endl;
}

void test_opengl(){
    using namespace Eigen;

    Vector3f x, y;
    x<<1,2,3;
    y<<3,4,5;
    Matrix3f rot = Matrix3f::Identity();

    Vector3f z = glVertex(y + rot * x);
    std::cout<<z<<std::endl;

    Quaterniond q;
    glRotate(q);
}

int main(){
    test_opengl();
    // test_allocator();
    // test_tensor();
    return 0;
}
