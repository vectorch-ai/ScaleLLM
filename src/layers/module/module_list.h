// Adapted from
// https://github.com/pytorch/pytorch/blob/main/torch/csrc/api/include/torch/nn/modules/container/modulelist.h
#pragma once

#include <utility>
#include <vector>

#include "module.h"
#include "module_holder.h"

namespace llm {
namespace detail {
/// A type trait whose `value` member is true if `M` derives from `Module`.
template <typename M>
using is_module = std::is_base_of<llm::Module, std::decay_t<M>>;

template <typename M, typename T = void>
using enable_if_module_t = std::enable_if_t<is_module<M>::value, T>;

}  // namespace detail

/// A list of `Module`s that registers its elements.
class ModuleListImpl : public Module {
 public:
  using Iterator = std::vector<std::shared_ptr<Module>>::iterator;
  using ConstIterator = std::vector<std::shared_ptr<Module>>::const_iterator;

  ModuleListImpl() = default;

  /// Constructs the `ModuleList` from a variadic list of modules.
  template <typename... Modules>
  explicit ModuleListImpl(Modules&&... modules) {
    modules_.reserve(sizeof...(Modules));
    push_back_var(std::forward<Modules>(modules)...);
  }

  /// Pretty prints the `ModuleList` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override {
    stream << "ModuleList";
  }

  void push_back(std::shared_ptr<Module> module) {
    modules_.push_back(std::move(module));
    const auto index = modules_.size() - 1;
    register_module(std::to_string(index), modules_[index]);
  }

  /// Adds a new `Module` to the `ModuleList` container, moving or copying
  /// it into a `shared_ptr` internally. This method allows passing value types,
  /// and letting the container deal with the boxing.
  template <typename M, typename = detail::enable_if_module_t<M>>
  void push_back(M&& module) {
    using Type = std::remove_reference_t<M>;
    push_back(std::make_shared<Type>(std::forward<M>(module)));
  }

  /// Unwraps the contained module of a `ModuleHolder` and adds it to the
  /// `ModuleList`.
  template <typename M>
  void push_back(const ModuleHolder<M>& module_holder) {
    push_back(module_holder.ptr());
  }

  /// Iterates over the container and calls `push_back()` on each value.
  template <typename Container>
  void extend(const Container& container) {
    for (const auto& module : container) {
      push_back(module);
    }
  }

  /// Returns an iterator to the start of the `ModuleList`.
  Iterator begin() { return modules_.begin(); }

  /// Returns a const iterator to the start of the `ModuleList`.
  ConstIterator begin() const { return modules_.begin(); }

  /// Returns an iterator to the end of the `ModuleList`.
  Iterator end() { return modules_.end(); }

  /// Returns a const iterator to the end of the `ModuleList`.
  ConstIterator end() const { return modules_.end(); }

  /// Attempts to return the module at the given index as the requested type.
  /// Throws an exception if the index is out of bounds or the types do not
  /// match.
  template <typename T>
  T& at(size_t index) {
    static_assert(detail::is_module<T>::value,
                  "Can only call ModuleList::at with an nn::Module type");
    TORCH_CHECK(index < size(), "Index out of range");
    auto module = modules_[index]->as<T>();
    TORCH_CHECK(module,
                "Unable to cast module[",
                index,
                "] to ",
                c10::demangle(typeid(T).name()));
    return *module;
  }

  /// Attempts to return the module at the given index as the requested type.
  /// Throws an exception if the index is out of bounds or the types do not
  /// match.
  template <typename T>
  const T& at(size_t index) const {
    static_assert(detail::is_module<T>::value,
                  "Can only call ModuleList::at with an nn::Module type");
    TORCH_CHECK(index < size(), "Index out of range");
    const auto module = modules_[index]->as<T>();
    TORCH_CHECK(module,
                "Unable to cast module[",
                index,
                "] to ",
                c10::demangle(typeid(T).name()));
    return *module;
  }

  /// Attempts to return a `std::shared_ptr` whose dynamic type is that of the
  /// underlying module at the given index. Throws an exception if the index is
  /// out of bounds.
  std::shared_ptr<Module> ptr(size_t index) const {
    TORCH_CHECK(index < size(), "Index out of range");
    return modules_[index];
  }

  /// Attempts to return a `std::shared_ptr` whose type is the one provided.
  /// Throws an exception if the index is out of bounds or the types do not
  /// match.
  template <typename T>
  std::shared_ptr<T> ptr(size_t index) const {
    static_assert(detail::is_module<T>::value,
                  "Can only call ModuleList::ptr with an nn::Module type");
    TORCH_CHECK(index < size(), "Index out of range");
    return std::dynamic_pointer_cast<T>(modules_[index]);
  }

  /// Like `ptr(index)`.
  std::shared_ptr<Module> operator[](size_t index) const {
    // This is the only method we can call without a type.
    return ptr(index);
  }

  /// The current size of the `ModuleList` container.
  size_t size() const noexcept { return modules_.size(); }

  /// True if there are no modules in the `ModuleList`.
  bool is_empty() const noexcept { return size() == 0; }

 private:
  template <typename Head, typename... Tail>
  void push_back_var(Head&& head, Tail&&... tail) {
    push_back(std::forward<Head>(head));
    // Recursively calls this method, until the parameter pack only thas this
    // entry left. Then calls `push_back()` a final time (above).
    push_back_var(std::forward<Tail>(tail)...);
  }

  /// The base case, when the list of modules is empty.
  void push_back_var() {}

  // Box the AnyModules to give ModuleList reference semantics, like the rest of
  // the API. Note that this is not required otherwise, this could just be a
  // `vector<AnyModule>`.
  std::vector<std::shared_ptr<Module>> modules_;
};

/// A `ModuleHolder` subclass for `ModuleListImpl`.
/// See the documentation for `ModuleListImpl` class to learn what methods it
/// provides, or the documentation for `ModuleHolder` to learn about PyTorch's
/// module storage semantics.
LLM_MODULE(ModuleList);

}  // namespace llm
