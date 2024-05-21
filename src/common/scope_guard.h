#pragma once
#include <type_traits>
#include <utility>

namespace llm {

// RAII object that invoked a callback on destruction.
template <typename Fun>
class ScopeGuard final {
 public:
  template <typename FuncArg>
  ScopeGuard(FuncArg&& f) : callback_(std::forward<FuncArg>(f)) {}

  // disallow copy and move
  ScopeGuard(const ScopeGuard&) = delete;
  ScopeGuard(ScopeGuard&&) = delete;
  ScopeGuard& operator=(const ScopeGuard&) = delete;
  ScopeGuard& operator=(ScopeGuard&&) = delete;

  ~ScopeGuard() noexcept {
    if (!dismissed_) {
      callback_();
    }
  }

  void dismiss() noexcept { dismissed_ = true; }

 private:
  Fun callback_;
  bool dismissed_ = false;
};

// allow function-to-pointer implicit conversions
template <typename Fun>
ScopeGuard(Fun&&) -> ScopeGuard<std::decay_t<Fun>>;

}  // namespace llm
