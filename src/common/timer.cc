#include "timer.h"

namespace neat {

Timer::Timer() { Reset(); }

void Timer::Reset() {
    current_duration_ = duration(0.0);
    total_duration_ = duration(0.0);
}

void Timer::Start() { begin_ = std::chrono::high_resolution_clock::now(); }

void Timer::Stop() {
    end_ = std::chrono::high_resolution_clock::now();
    current_duration_ = end_ - begin_;
    total_duration_ += current_duration_;
}

double Timer::GetCurrentDuration() const { return current_duration_.count(); }

double Timer::GetTotalDuration() const { return total_duration_.count(); }

}  // namespace neat
