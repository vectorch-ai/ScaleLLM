#pragma once

#include <torch/arg.h>
#include <torch/csrc/utils/variadic.h>
#include <torch/detail/static.h>
#include <torch/serialize/archive.h>
#include <torch/types.h>

#include <memory>
#include <type_traits>
#include <utility>

namespace llm {
namespace detail {
// This class exists  only to do SFINAE on abstract types `T` that are really
// `ModuleHolder<ModuleType>`, because there's no good way to say that `T` is a
// `ModuleHolder` over some unknown type `ModuleType`. With this, you can do
// `enable_if_t<is_base_of_v<ModuleHolderIndicator, T>>`.
struct ModuleHolderIndicator {};

// A type trait that is true for types that are `ModuleHolder`s.
template <typename T>
using is_module_holder =
    std::is_base_of<ModuleHolderIndicator, std::decay_t<T>>;

template <typename T>
using disable_if_module_holder_t =
    std::enable_if_t<!is_module_holder<T>::value>;

// A collection of templates that answer the question whether a type `T` is a
// `ModuleHolder`, and if so whether its contained type is of type `C`. This is
// tricky because it is hard to short circuit in template metaprogramming. A
// naive and incorrect solution to this problem would be something like
// `disable_if<is_module_holder<T>::value && typename T::ContainedType == C>`.
// This would disable all types that are not `ModuleHolder`s, because even
// though the `is_module_holder<T>::value` may be `false` for such types the
// `T::ContainedType` access would be ill-formed and thus fail the whole
// expression by the rules of SFINAE. Instead we have to use template
// specialization to statically branch on the first condition
// (`is_module_holder<T>`) and are only then allowed to query
// `T::ContainedType` in the branch for which the condition was true.

// Base template.
template <bool is_module_holder_value, typename T, typename C>
struct is_module_holder_of_impl;

// False branch. `T` is not a `ModuleHolder` and thus not a `ModuleHolder` with
// contained type `C`.
template <typename T, typename C>
struct is_module_holder_of_impl<false, T, C> : std::false_type {};

// True branch. `T` is a `ModuleHolder` and thus we can legit access its
// `ContainedType` and compare it against `C`.
template <typename T, typename C>
struct is_module_holder_of_impl<true, T, C>
    : std::is_same<typename T::ContainedType, C> {};

// Helper template.
template <typename T, typename C>
struct is_module_holder_of
    : is_module_holder_of_impl<is_module_holder<T>::value,
                               std::decay_t<T>,
                               std::decay_t<C>> {};

// A collection of templates that allow deducing the return type of the
// `forward()` method, but only if a module actually has a `forward()` method,
// and otherwise deduces to the type `void`.

template <bool has_forward_value, typename C, typename... Args>
struct return_type_of_forward_impl;

template <typename C, typename... Args>
struct return_type_of_forward_impl<true, C, Args...> {
  using type = decltype(::std::declval<C>().forward(::std::declval<Args>()...));
};

template <typename C, typename... Args>
struct return_type_of_forward_impl<false, C, Args...> {
  using type = void;
};

template <typename C, typename... Args>
using return_type_of_forward =
    return_type_of_forward_impl<torch::detail::has_forward<C>::value,
                                C,
                                Args...>;

template <typename C, typename... Args>
using return_type_of_forward_t =
    typename return_type_of_forward<C, Args...>::type;
}  // namespace detail

namespace nn {
using namespace torch;

/// A `ModuleHolder` is essentially a wrapper around `std::shared_ptr<M>` where
/// `M` is an `nn::Module` subclass, with convenient constructors defined for
/// the kind of constructions we want to allow for our modules.
template <typename Contained>
class ModuleHolder : detail::ModuleHolderIndicator {
 protected:
  /// The module pointer this class wraps.
  /// NOTE: Must be placed at the top of the class so that we can use it with
  /// trailing return types below.
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::shared_ptr<Contained> impl_;

 public:
  using ContainedType = Contained;

  /// Default constructs the contained module if if has a default constructor,
  /// else produces a static error.
  ///
  /// NOTE: This uses the behavior of template
  /// classes in C++ that constructors (or any methods) are only compiled when
  /// actually used.
  ModuleHolder() : impl_(default_construct()) {
    static_assert(
        std::is_default_constructible_v<Contained>,
        "You are trying to default construct a module which has "
        "no default constructor. Use = nullptr to give it the empty state "
        "(e.g. `Linear linear = nullptr;` instead of `Linear linear;`).");
  }

  /// Constructs the `ModuleHolder` with an empty contained value. Access to
  /// the underlying module is not permitted and will throw an exception, until
  /// a value is assigned.
  /* implicit */ ModuleHolder(std::nullptr_t) : impl_(nullptr) {}

  /// Constructs the `ModuleHolder` with a contained module, forwarding all
  /// arguments to its constructor.
  template <typename Head,
            typename... Tail,
            typename = std::enable_if_t<
                !(detail::is_module_holder_of<Head, ContainedType>::value &&
                  (sizeof...(Tail) == 0))>>
  explicit ModuleHolder(Head&& head, Tail&&... tail)
      : impl_(new Contained(std::forward<Head>(head),
                            std::forward<Tail>(tail)...)) {}

  /// Constructs the `ModuleHolder` from a pointer to the contained type.
  /// Example: `Linear(std::make_shared<LinearImpl>(...))`.
  /* implicit */ ModuleHolder(std::shared_ptr<Contained> module)
      : impl_(std::move(module)) {}

  /// Returns true if the `ModuleHolder` contains a module, or false if it is
  /// `nullptr`.
  explicit operator bool() const noexcept { return !is_empty(); }

  /// Forwards to the contained module.
  Contained* operator->() { return get(); }

  /// Forwards to the contained module.
  const Contained* operator->() const { return get(); }

  /// Returns a reference to the contained module.
  Contained& operator*() { return *get(); }

  /// Returns a const reference to the contained module.
  const Contained& operator*() const { return *get(); }

  /// Returns a shared pointer to the underlying module.
  const std::shared_ptr<Contained>& ptr() const {
    TORCH_CHECK(!is_empty(), "Accessing empty ModuleHolder");
    return impl_;
  }

  /// Returns a pointer to the underlying module.
  Contained* get() {
    TORCH_CHECK(!is_empty(), "Accessing empty ModuleHolder");
    return impl_.get();
  }

  /// Returns a const pointer to the underlying module.
  const Contained* get() const {
    TORCH_CHECK(!is_empty(), "Accessing empty ModuleHolder");
    return impl_.get();
  }

  /// Calls the `forward()` method of the contained module.
  template <typename... Args>
  auto operator()(Args&&... args)
      -> detail::return_type_of_forward_t<Contained, Args...> {
    // This will not compile if the module does not have a `forward()` method
    // (as expected).
    // NOTE: `std::forward` is qualified to prevent VS2017 emitting
    // error C2872: 'std': ambiguous symbol
    return impl_->forward(::std::forward<Args>(args)...);
  }

  /// Forwards to the subscript operator of the contained module.
  /// NOTE: std::forward is qualified to prevent VS2017 emitting
  ///       error C2872: 'std': ambiguous symbol
  template <typename Arg>
  decltype(auto) operator[](Arg&& arg) {
    return (*impl_)[::std::forward<Arg>(arg)];
  }

  /// Returns true if the `ModuleHolder` does not contain a module.
  bool is_empty() const noexcept { return impl_ == nullptr; }

 private:
  template <typename T = Contained>
  std::shared_ptr<Contained> default_construct() {
    if constexpr (std::is_default_constructible_v<T>) {
      return std::make_shared<Contained>();
    } else {
      return nullptr;
    }
  }
};

/// Pretty prints the given `Module` into the `ostream`.
template <typename ModuleType>
std::ostream& operator<<(std::ostream& stream,
                         const nn::ModuleHolder<ModuleType>& module) {
  return stream << *module;
}

}  // namespace nn
}  // namespace llm

/// Defines a class `Name` which inherits from `nn::ModuleHolder` to provide a
/// wrapper over a `std::shared_ptr<ImplType>`.
/// `Impl` is a type alias for `ImplType` which provides a way to call static
/// method of `ImplType`.
#define LLM_MODULE_IMPL(Name, ImplType)                              \
  class Name : public llm::nn::ModuleHolder<ImplType> { /* NOLINT */ \
   public:                                                           \
    using llm::nn::ModuleHolder<ImplType>::ModuleHolder;             \
    using Impl [[maybe_unused]] = ImplType;                          \
  }

/// Like `LLM_MODULE_IMPL`, but defaults the `ImplType` name to `<Name>Impl`.
#define LLM_MODULE(Name) LLM_MODULE_IMPL(Name, Name##Impl)
