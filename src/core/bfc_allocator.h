#ifndef CORE_BFC_ALLOCATOR_H_
#define CORE_BFC_ALLOCATOR_H_

class BFCAllocator{
    public:
        BFCAllocator();
        virtual ~BFCAllocator();
        void Merge();


    private:
        static const int kNumBins = 21;
};

#endif
