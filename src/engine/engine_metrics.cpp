#include "engine_metrics.h"

DEFINE_COUNTER(prepare_input_latency_seconds,
               "Latency of preparing input in seconds");
DEFINE_COUNTER_FAMILY(execution_latency_seconds,
                      "Execution latency in seconds");
DEFINE_COUNTER_INSTANCE(model_execution_latency_seconds,
                        execution_latency_seconds,
                        {{"stage", "model"}});
DEFINE_COUNTER_INSTANCE(logits_processing_latency_seconds,
                        execution_latency_seconds,
                        {{"stage", "logits_processing"}});
DEFINE_COUNTER_INSTANCE(sampling_latency_seconds,
                        execution_latency_seconds,
                        {{"stage", "sampling"}});
