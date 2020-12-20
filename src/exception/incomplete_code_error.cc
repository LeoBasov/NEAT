#include "incomplete_code_error.h"

namespace neat {

IncompleteCodeError::IncompleteCodeError(const std::string &what) : Exception(what, "") {}

}  // namespace neat
