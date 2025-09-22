#pragma once

#include <ATen/ATen.h>
#include <torch/ordered_dict.h>
#include <torch/types.h>

#include <functional>
#include <iosfwd>
#include <memory>
#include <string>

#include "module_holder.h"

namespace llm::nn {

/// The base class for all modules.
///
/// A `Module` is an abstraction over the implementation of some function or
/// algorithm. A `Module` may contain further `Module`s ("submodules"), each
/// with their own implementation and further submodules. `Module`s can thus be
/// said to form a recursive tree structure. A `Module` is registered as a
/// submodule to another `Module` by calling `register_module()`, typically from
/// within a parent module's constructor.
class Module : public std::enable_shared_from_this<Module> {
 public:
  /// Tells the base `Module` about the name of the submodule.
  explicit Module(std::string name);

  /// Constructs the module without immediate knowledge of the submodule's name.
  /// The name of the submodule is inferred via RTTI (if possible) the first
  /// time `.name()` is invoked.
  Module();

  // default copy/move constructors and assignment operators
  Module(const Module&) = default;
  Module& operator=(const Module&) = default;
  Module(Module&&) noexcept = default;
  Module& operator=(Module&&) noexcept = default;

  virtual ~Module() = default;

  /// Returns the name of the `Module`.
  const std::string& name() const noexcept;

  /// Returns the parameters of this `Module` and if `recurse` is true, also
  /// recursively of every submodule.
  std::vector<torch::Tensor> parameters(bool recurse = true) const;

  /// Returns an `OrderedDict` with the parameters of this `Module` along with
  /// their keys, and if `recurse` is true also recursively of every submodule.
  torch::OrderedDict<std::string, torch::Tensor> named_parameters(
      bool recurse = true) const;

  /// Returns the buffers of this `Module` and if `recurse` is true, also
  /// recursively of every submodule.
  std::vector<torch::Tensor> buffers(bool recurse = true) const;

  /// Returns an `OrderedDict` with the buffers of this `Module` along with
  /// their keys, and if `recurse` is true also recursively of every submodule.
  torch::OrderedDict<std::string, torch::Tensor> named_buffers(
      bool recurse = true) const;

  /// Returns the submodules of this `Module` (the entire submodule hierarchy)
  /// and if `include_self` is true, also inserts a `shared_ptr` to this module
  /// in the first position.
  std::vector<std::shared_ptr<Module>> modules(bool include_self = true) const;

  /// Returns an `OrderedDict` of the submodules of this `Module` (the entire
  /// submodule hierarchy) and their keys, and if `include_self` is true, also
  /// inserts a `shared_ptr` to this module in the first position. If
  /// `name_prefix` is given, it is prepended to every key as
  /// `<name_prefix>.<key>` (and just `name_prefix` for the module itself).
  torch::OrderedDict<std::string, std::shared_ptr<Module>> named_modules(
      const std::string& name_prefix = std::string(),
      bool include_self = true) const;

  /// Returns the direct submodules of this `Module`.
  std::vector<std::shared_ptr<Module>> children() const;

  /// Returns an `OrderedDict` of the direct submodules of this `Module` and
  /// their keys.
  torch::OrderedDict<std::string, std::shared_ptr<Module>> named_children()
      const;

  /// Applies the `function` to the `Module` and recursively to every submodule.
  using ModuleApplyFunction = std::function<void(Module&)>;
  void apply(const ModuleApplyFunction& function);

  using ConstModuleApplyFunction = std::function<void(const Module&)>;
  void apply(const ConstModuleApplyFunction& function) const;

  using NamedModuleApplyFunction =
      std::function<void(const std::string&, Module&)>;
  void apply(const NamedModuleApplyFunction& function,
             const std::string& name_prefix = std::string());

  using ConstNamedModuleApplyFunction =
      std::function<void(const std::string&, const Module&)>;
  void apply(const ConstNamedModuleApplyFunction& function,
             const std::string& name_prefix = std::string()) const;

  using ModulePointerApplyFunction =
      std::function<void(const std::shared_ptr<Module>&)>;
  void apply(const ModulePointerApplyFunction& function) const;

  using NamedModulePointerApplyFunction =
      std::function<void(const std::string&, const std::shared_ptr<Module>&)>;
  void apply(const NamedModulePointerApplyFunction& function,
             const std::string& name_prefix = std::string()) const;

  /// Recursively casts all parameters to the given `dtype` and `device`.
  ///
  /// If `non_blocking` is true and the source is in pinned memory and
  /// destination is on the GPU or vice versa, the copy is performed
  /// asynchronously with respect to the host. Otherwise, the argument has no
  /// effect.
  virtual void to(torch::Device device,
                  torch::Dtype dtype,
                  bool non_blocking = false);

  /// Recursively casts all parameters to the given dtype.
  virtual void to(torch::Dtype dtype, bool non_blocking = false);

  /// Recursively moves all parameters to the given device.
  virtual void to(torch::Device device, bool non_blocking = false);

  /// Attempts to cast this `Module` to the given `ModuleType`.
  template <typename ModuleType>
  typename ModuleType::ContainedType* as() noexcept;

  template <typename ModuleType>
  const typename ModuleType::ContainedType* as() const noexcept;

  template <typename ModuleType,
            typename = detail::disable_if_module_holder_t<ModuleType>>
  ModuleType* as() noexcept;

  template <typename ModuleType,
            typename = detail::disable_if_module_holder_t<ModuleType>>
  const ModuleType* as() const noexcept;

  /// Streams a pretty representation of the `Module` into the given `stream`.
  /// By default, this representation will be the name of the module (taken from
  /// `name()`), followed by a recursive pretty print of all of the `Module`'s
  /// submodules.
  ///
  /// Override this method to change the pretty print. The input
  /// `stream` should be returned from the method, to allow easy chaining.
  virtual void pretty_print(std::ostream& stream) const;

  /// Registers a parameter with this `Module`.
  torch::Tensor& register_parameter(std::string name, torch::Tensor tensor);

  /// Registers a buffer with this `Module`.
  torch::Tensor& register_buffer(std::string name, torch::Tensor tensor);

  /// Registers a submodule with this `Module`.
  template <typename ModuleType>
  std::shared_ptr<ModuleType> register_module(
      std::string name,
      std::shared_ptr<ModuleType> module);

  template <typename ModuleType>
  std::shared_ptr<ModuleType> register_module(
      std::string name,
      ModuleHolder<ModuleType> module_holder);

  /// Replaces a registered submodule with this `Module`.
  template <typename ModuleType>
  std::shared_ptr<ModuleType> replace_module(
      const std::string& name,
      std::shared_ptr<ModuleType> module);

  template <typename ModuleType>
  std::shared_ptr<ModuleType> replace_module(
      const std::string& name,
      ModuleHolder<ModuleType> module_holder);

  /// Unregisters a submodule from this `Module`. If there is no such module
  /// with `name` an exception is thrown.
  void unregister_module(const std::string& name);

 protected:
  /// The registered parameters of this `Module`.
  /// Inorder to access parameters_ in ParameterDict and ParameterList
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  torch::OrderedDict<std::string, torch::Tensor> parameters_;

 private:
  /// Pretty prints the given `Module` into the `ostream`.
  friend std::ostream& operator<<(std::ostream& stream,
                                  const nn::Module& module);

  // Private methods.

  /// The implementation of the various `to()` methods.
  template <typename... Ts>
  void to_impl(Ts&&... ts);

  /// Implements pretty printing the module hierarchy.
  void pretty_print_recursive(std::ostream& stream,
                              const std::string& indentation) const;

  /// Applies the `function` to every submodule recursively, starting at this
  /// `Module`'s children (thus not including the module itself).
  void apply_to_submodules(
      const NamedModulePointerApplyFunction& function,
      const std::string& name_prefix = std::string()) const;

  /// Returns a shared_ptr to `this` in a safe (checked) way.
  std::shared_ptr<Module> shared_from_this_checked() const;

  /// The registered buffers of this `Module`.
  torch::OrderedDict<std::string, torch::Tensor> buffers_;

  /// The registered (direct) submodules of this `Module`.
  torch::OrderedDict<std::string, std::shared_ptr<Module>> children_;

  /// The module's name (e.g. "LSTM").
  mutable std::optional<std::string> name_;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ nn::Module ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <typename ModuleType>
typename ModuleType::ContainedType* Module::as() noexcept {
  // Use the contained type of the `ModuleHolder`, e.g. `LinearImpl` for
  // `Linear`, since `LinearImpl` inherits `nn::Module`.
  return as<typename ModuleType::ContainedType>();
}

template <typename ModuleType>
const typename ModuleType::ContainedType* Module::as() const noexcept {
  // Use the contained type of the `ModuleHolder`, e.g. `LinearImpl` for
  // `Linear`, since `LinearImpl` inherits `nn::Module`.
  return as<typename ModuleType::ContainedType>();
}

template <typename ModuleType, typename>
ModuleType* Module::as() noexcept {
  return dynamic_cast<ModuleType*>(this);
}

template <typename ModuleType, typename>
const ModuleType* Module::as() const noexcept {
  return dynamic_cast<const ModuleType*>(this);
}

template <typename ModuleType>
std::shared_ptr<ModuleType> Module::register_module(
    std::string name,
    std::shared_ptr<ModuleType> module) {
  TORCH_CHECK(!name.empty(), "Submodule name must not be empty");
  TORCH_CHECK(name.find('.') == std::string::npos,
              "Submodule name must not contain a dot (got '",
              name,
              "')");
  auto& base_module = children_.insert(std::move(name), std::move(module));
  return std::dynamic_pointer_cast<ModuleType>(base_module);
}

template <typename ModuleType>
std::shared_ptr<ModuleType> Module::register_module(
    std::string name,
    ModuleHolder<ModuleType> module_holder) {
  return register_module(std::move(name), module_holder.ptr());
}

template <typename ModuleType>
std::shared_ptr<ModuleType> Module::replace_module(
    const std::string& name,
    std::shared_ptr<ModuleType> module) {
  auto& base_module = (children_[name] = std::move(module));
  return std::dynamic_pointer_cast<ModuleType>(base_module);
}

template <typename ModuleType>
std::shared_ptr<ModuleType> Module::replace_module(
    const std::string& name,
    ModuleHolder<ModuleType> module_holder) {
  return replace_module(name, module_holder.ptr());
}

template <typename... Ts>
void Module::to_impl(Ts&&... ts) {
  // First call `to()` on every child module.
  for (auto& child : children_) {
    child.value()->to(ts...);
  }
  // Then move every parameter to the new dtype/device.
  for (auto& parameter : named_parameters(/*recurse=*/false)) {
    parameter->set_data(parameter->to(ts...));
  }
  // Then move every buffer to the new dtype/device.
  for (auto& buffer : named_buffers(/*recurse=*/false)) {
    buffer->set_data(buffer->to(ts...));
  }
}

}  // namespace llm::nn
