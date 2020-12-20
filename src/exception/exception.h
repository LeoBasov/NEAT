#pragma once

#include <exception>
#include <string>

namespace neat {
class Exception : public std::exception {
   public:
    Exception(const std::string& what, const std::string& where = "");
    virtual ~Exception() = default;

    virtual const char* what() const throw();
    std::string where() const throw();

   private:
    std::string what_;
    std::string where_;
};
}  // namespace neat
