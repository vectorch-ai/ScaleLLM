#pragma once

#include <utility>

#define DEFINE_ARG(T, name)                                \
 public:                                                   \
  inline auto name(T name)->decltype(*this) { /* NOLINT */ \
    this->name##_ = name;                                  \
    return *this;                                          \
  }                                                        \
  inline const T& name() const noexcept { /* NOLINT */     \
    return this->name##_;                                  \
  }                                                        \
  inline T& name() noexcept { /* NOLINT */                 \
    return this->name##_;                                  \
  }                                                        \
                                                           \
 private:                                                  \
  T name##_ /* NOLINT */

#define DEFINE_PTR_ARG(T, name)                             \
 public:                                                    \
  inline auto name(T* name)->decltype(*this) { /* NOLINT */ \
    this->name##_ = name;                                   \
    return *this;                                           \
  }                                                         \
  inline T* name() const noexcept { /* NOLINT */            \
    return this->name##_;                                   \
  }                                                         \
                                                            \
 private:                                                   \
  T* name##_ /* NOLINT */
