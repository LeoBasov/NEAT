#pragma once

#include "exception.h"

namespace neat {
class IncompleteCodeError : public Exception {
   public:
    IncompleteCodeError(const std::string& what);
    ~IncompleteCodeError() = default;
};
}  // namespace neat
