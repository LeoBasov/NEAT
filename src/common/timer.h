#pragma once

#include <chrono>

namespace neat {
class Timer {
    using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;
    using duration = std::chrono::duration<double>;

   public:
    Timer();
    ~Timer() = default;

    void Reset();
    void Start();
    void Stop();

    double GetCurrentDuration() const;
    double GetTotalDuration() const;

   private:
    time_point begin_;
    time_point end_;
    duration current_duration_;
    duration total_duration_;
};
}  // namespace neat
