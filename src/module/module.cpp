#include "module.h"

#include <typeinfo>

#include "model_loader/state_dict.h"

namespace llm {
using namespace torch;

namespace {
/// Joins names hierarchically: "name_prefix.name" if `name_prefix` is
/// non-empty, else just "name".
std::string join_name(const std::string& name_prefix, const std::string& name) {
  size_t total_size = name.size();
  if (!name_prefix.empty()) {
    total_size += name_prefix.size() + 1;
  }
  std::string full_name;
  full_name.reserve(total_size);
  if (!name_prefix.empty()) {
    full_name += name_prefix;
    full_name.push_back('.');
  }
  full_name += name;
  return full_name;
}
}  // namespace

Module::Module()
    : parameters_("Parameter"), buffers_("Buffer"), children_("Submodule") {}

Module::Module(std::string name) : Module() {
  // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
  name_ = std::move(name);
}

const std::string& Module::name() const noexcept {
  // If the name optional is empty at this point, we grab the name of the
  // dynamic type via RTTI.
  if (!name_.has_value()) {
    name_ = c10::demangle(typeid(*this).name());
  }
  return *name_;
}

std::vector<Tensor> Module::parameters(bool recurse) const {
  return named_parameters(recurse).values();
}

OrderedDict<std::string, Tensor> Module::named_parameters(bool recurse) const {
  OrderedDict<std::string, Tensor> result;
  if (!recurse) {
    for (const auto& parameter : parameters_) {
      const auto& param_tensor = parameter.value().tensor;
      if (param_tensor.defined()) {
        result.insert(parameter.key(), param_tensor);
      }
    }
  } else {
    apply([&result](const std::string& name, const Module& module) {
      for (const auto& parameter : module.named_parameters(/*recurse=*/false)) {
        TORCH_INTERNAL_ASSERT(parameter.value().defined());
        result.insert(join_name(name, parameter.key()), parameter.value());
      }
    });
  }
  return result;
}

std::vector<Tensor> Module::buffers(bool recurse) const {
  return named_buffers(recurse).values();
}

OrderedDict<std::string, Tensor> Module::named_buffers(bool recurse) const {
  OrderedDict<std::string, Tensor> result;
  if (!recurse) {
    for (const auto& buffer : buffers_) {
      if (buffer.value().defined()) {
        result.insert(buffer.key(), buffer.value());
      }
    }
  } else {
    apply([&result](const std::string& name, const Module& module) {
      for (const auto& buffer : module.named_buffers(/*recurse=*/false)) {
        TORCH_INTERNAL_ASSERT(buffer.value().defined());
        result.insert(join_name(name, buffer.key()), buffer.value());
      }
    });
  }
  return result;
}

std::vector<std::shared_ptr<Module>> Module::modules(bool include_self) const {
  std::vector<std::shared_ptr<Module>> result;
  if (include_self) {
    apply([&result](const std::shared_ptr<Module>& module) {
      result.push_back(module);
    });
  } else {
    apply_to_submodules(
        [&result](const std::string&, const std::shared_ptr<Module>& module) {
          result.push_back(module);
        });
  }
  return result;
}

OrderedDict<std::string, std::shared_ptr<Module>> Module::named_modules(
    const std::string& name_prefix,
    bool include_self) const {
  OrderedDict<std::string, std::shared_ptr<Module>> result;
  if (include_self) {
    apply(
        [&result](const std::string& key,
                  const std::shared_ptr<Module>& module) {
          result.insert(key, module);
        },
        name_prefix);
  } else {
    apply_to_submodules(
        [&result](const std::string& key,
                  const std::shared_ptr<Module>& module) {
          result.insert(key, module);
        },
        name_prefix);
  }
  return result;
}

std::vector<std::shared_ptr<Module>> Module::children() const {
  std::vector<std::shared_ptr<Module>> result;
  for (const auto& item : children_) {
    result.push_back(item.value().module);
  }
  return result;
}

OrderedDict<std::string, std::shared_ptr<Module>> Module::named_children()
    const {
  OrderedDict<std::string, std::shared_ptr<Module>> result;
  for (const auto& item : children_) {
    result.insert(item.key(), item.value().module);
  }
  return result;
}

void Module::apply(const ModuleApplyFunction& function) {
  function(*this);
  apply_to_submodules(
      [&function](const std::string&, const std::shared_ptr<Module>& module) {
        function(*module);
      });
}

void Module::apply(const ConstModuleApplyFunction& function) const {
  function(*this);
  apply_to_submodules(
      [&function](const std::string&, const std::shared_ptr<Module>& module) {
        function(*module);
      });
}

void Module::apply(const NamedModuleApplyFunction& function,
                   const std::string& name_prefix) {
  function(/*name=*/name_prefix, *this);
  apply_to_submodules(
      [&function](const std::string& name,
                  const std::shared_ptr<Module>& module) {
        function(name, *module);
      },
      name_prefix);
}

void Module::apply(const ConstNamedModuleApplyFunction& function,
                   const std::string& name_prefix) const {
  function(/*name=*/name_prefix, *this);
  apply_to_submodules(
      [&function](const std::string& name,
                  const std::shared_ptr<Module>& module) {
        function(name, *module);
      },
      name_prefix);
}

void Module::apply(const ModulePointerApplyFunction& function) const {
  function(shared_from_this_checked());
  apply_to_submodules(
      [&function](const std::string&, const std::shared_ptr<Module>& module) {
        function(module);
      });
}

void Module::apply(const NamedModulePointerApplyFunction& function,
                   const std::string& name_prefix) const {
  function(
      /*name=*/name_prefix, shared_from_this_checked());
  apply_to_submodules(function, name_prefix);
}

void Module::to(torch::Device device, torch::Dtype dtype, bool non_blocking) {
  to_impl(device, dtype, non_blocking);
}

void Module::to(torch::Dtype dtype, bool non_blocking) {
  to_impl(dtype, non_blocking);
}

void Module::to(torch::Device device, bool non_blocking) {
  to_impl(device, non_blocking);
}

Tensor& Module::register_parameter(std::string name,
                                   Tensor tensor,
                                   const TensorLoader& loader) {
  TORCH_CHECK(!name.empty(), "Parameter name must not be empty");
  // tensor.set_requires_grad(false);
  Parameter param{
      .tensor = std::move(tensor), .loader = loader, .is_loaded = false};
  return parameters_.insert(std::move(name), std::move(param)).tensor;
}

Tensor& Module::register_parameter(std::string name, Tensor tensor) {
  auto default_loader = [](const StateDict& sd, const std::string& key) {
    return sd.get_tensor(key);
  };
  return register_parameter(std::move(name), std::move(tensor), default_loader);
}

Tensor& Module::register_sharded_parameter(std::string name,
                                           int dim,
                                           int rank,
                                           int world_size,
                                           Tensor tensor) {
  TORCH_CHECK(!name.empty(), "Parameter name must not be empty");
  // tensor.set_requires_grad(false);
  Parameter param{
      .tensor = std::move(tensor),
      .loader =
          [dim, rank, world_size](const StateDict& sd, const std::string& key) {
            return sd.get_sharded_tensor(key, dim, rank, world_size);
          },
      .is_loaded = false};
  return parameters_.insert(std::move(name), std::move(param)).tensor;
}

Tensor& Module::register_buffer(std::string name, Tensor tensor) {
  TORCH_CHECK(!name.empty(), "Buffer name must not be empty");
  return buffers_.insert(std::move(name), std::move(tensor));
}

bool Module::unregister_module(const std::string& name) {
  if (!children_.contains(name)) {
    return false;
  }
  children_.erase(name);
  return true;
}

void Module::pretty_print(std::ostream& stream) const { stream << name(); }

void Module::pretty_print_recursive(std::ostream& stream,
                                    const std::string& indentation) const {
  pretty_print(stream);
  if (!children_.is_empty()) {
    stream << "(\n";
    const std::string next_indentation = indentation + "  ";
    for (const auto& child : children_) {
      stream << next_indentation << "(" << child.key() << "): ";
      child.value().module->pretty_print_recursive(stream, next_indentation);
      stream << '\n';
    }
    stream << indentation << ")";
  }
}

// NOLINTNEXTLINE(misc-no-recursion)
void Module::apply_to_submodules(
    const NamedModulePointerApplyFunction& function,
    const std::string& name_prefix) const {
  for (const auto& child : children_) {
    auto qualified_name = join_name(name_prefix, child.key());
    function(qualified_name, child.value().module);
    child.value().module->apply_to_submodules(function, qualified_name);
  }
}

std::shared_ptr<Module> Module::shared_from_this_checked() const {
  std::shared_ptr<const Module> ptr;
  try {
    ptr = shared_from_this();
  } catch (const std::bad_weak_ptr&) {
    TORCH_CHECK(
        false,
        "It looks like you attempted to retrieve your top-level module "
        "as a shared_ptr, but it is not stored in a shared_ptr. "
        "Use std::make_shared<",
        name(),
        "> instead of creating your module on "
        "the stack, or alternatively do not try to access your top-level "
        "module at all by passing /*include_self=*/false "
        "to modules() or named_modules()");
  }
  return std::const_pointer_cast<Module>(ptr);
}

std::ostream& operator<<(std::ostream& stream, const Module& module) {
  module.pretty_print_recursive(stream, "");
  return stream;
}

// load weights from the checkpoint, override this method if necessary
// NOLINTNEXTLINE(misc-no-recursion)
size_t Module::load(const StateDict& state_dict,
                    const std::string& name_prefix) {
  size_t total_loaded = 0;
  // load parameters one by one
  for (auto& item : parameters_) {
    const auto& key = item.key();
    auto& param = item.value();
    const auto& param_tensor = param.tensor;
    // skip loading for undefined tensor
    if (!param_tensor.defined()) {
      // mark as loaded to pass verification
      param.is_loaded = true;
      continue;
    }

    const auto tensor = param.loader(state_dict, key);
    if (!tensor.defined()) {
      continue;
    }

    if (param.is_loaded) {
      LOG(WARNING) << "Parameter " << join_name(name_prefix, key)
                   << " is already loaded";
    }

    if (param_tensor.sizes() == tensor.sizes()) {
      LOG(INFO) << "Loading parameter: " << join_name(name_prefix, key)
                << " of size " << tensor.sizes();
      // copy data to the parameter tensor
      param_tensor.copy_(tensor);
      // mark as loaded
      param.is_loaded = true;
      ++total_loaded;
    } else {
      LOG(ERROR) << "Size mismatch for parameter "
                 << join_name(name_prefix, key) << ": expected "
                 << param_tensor.sizes() << ", got " << tensor.sizes();
    }
  }

  // don't need to load buffers, since they are initialized in the constructor

  // recursively load children modules
  for (const auto& item : children_) {
    const auto& key = item.key();
    const auto& child = item.value();
    if (child.selector) {
      // select state dict for the child module
      const auto child_state_dict = child.selector(state_dict, key);
      total_loaded +=
          child.module->load(child_state_dict, join_name(name_prefix, key));
    } else {
      total_loaded += child.module->load(state_dict, name_prefix);
    }
  }
  return total_loaded;
}

// verify whether the weights are loaded, override this method if necessary
// NOLINTNEXTLINE(misc-no-recursion)
bool Module::verify(const std::string& name_prefix) const {
  bool all_loaded = true;
  for (const auto& item : parameters_) {
    const auto& key = item.key();
    const auto& param = item.value();
    if (!param.is_loaded) {
      LOG(ERROR) << "Missing parameter: " << join_name(name_prefix, key);
    }
    all_loaded = all_loaded && param.is_loaded;
  }

  for (const auto& item : children_) {
    const auto& key = item.key();
    const auto& child = item.value();
    const std::string prefix =
        child.selector ? join_name(name_prefix, key) : name_prefix;
    const bool child_loaded = child.module->verify(prefix);
    all_loaded = all_loaded && child_loaded;
  }
  return all_loaded;
}
}  // namespace llm
