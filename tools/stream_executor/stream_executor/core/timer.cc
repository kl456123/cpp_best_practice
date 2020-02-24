#include "stream_executor/core/timer.h"
#include "stream_executor/core/stream_executor_pimpl.h"


Timer::Timer(StreamExecutor *parent): parent_(parent),
    implementation_(parent_->implementation()->GetTimerImplementation()) {}

Timer::~Timer() { parent_->DeallocateTimer(this); }

uint64 Timer::Microseconds() const { return implementation_->Microseconds(); }

uint64 Timer::Nanoseconds() const { return implementation_->Nanoseconds(); }
