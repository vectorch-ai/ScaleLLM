#pragma once

namespace llm {
// a central place to define common macros for the project
// clang-format off
#define DEFINE_ARG(T, name)                                       \
 public:                                                          \
  inline auto name(const T& name) ->decltype(*this) {             \
    this->name##_ = name;                                         \
    return *this;                                                 \
  }                                                               \
  inline const T& name() const noexcept { return this->name##_; } \
  inline T& name() noexcept { return this->name##_; }             \
                                                                  \
  T name##_

#define DEFINE_PTR_ARG(T, name)                             \
 public:                                                    \
  inline auto name(T* name) ->decltype(*this) {             \
    this->name##_ = name;                                   \
    return *this;                                           \
  }                                                         \
  inline T* name() const noexcept { return this->name##_; } \
                                                            \
  T* name##_

// clang-format on

#ifndef UNUSED_PARAMETER
#define UNUSED_PARAMETER(x) ((void)(x))
#endif

#if __has_attribute(guarded_by)
#define GUARDED_BY(x) __attribute__((guarded_by(x)))
#else
#define GUARDED_BY(x)
#endif

// concatenate two strings
#define LLM_STR_CAT(s1, s2) s1##s2

// create an anonymous variable
#define LLM_ANON_VAR(str) LLM_STR_CAT(str, __LINE__)

#define REQUIRES(...) std::enable_if_t<(__VA_ARGS__)>* = nullptr

}  // namespace llm