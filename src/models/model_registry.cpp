#include "model_registry.h"

#include <glog/logging.h>

// list all registered models here
#include "huggingface/aquila.h"
#include "huggingface/bloom.h"
#include "huggingface/gpt2.h"
#include "huggingface/gpt_j.h"
#include "huggingface/gpt_neox.h"
#include "huggingface/internlm.h"
#include "huggingface/llama.h"
#include "huggingface/mistral.h"
#include "huggingface/mpt.h"
#include "llama.h"

namespace llm {

ModelRegistry* ModelRegistry::get_instance() {
  static ModelRegistry registry;
  return &registry;
}

}  // namespace llm
