#include "handler_metrics.h"

DEFINE_COUNTER_FAMILY(request_status_total, "Total number of request status");
DEFINE_COUNTER_INSTANCE(request_ok, request_status_total, {{"code", "OK"}});
DEFINE_COUNTER_INSTANCE(request_cancelled,
                        request_status_total,
                        {{"code", "CANCELLED"}});
DEFINE_COUNTER_INSTANCE(request_unknown,
                        request_status_total,
                        {{"code", "UNKNOWN"}});
DEFINE_COUNTER_INSTANCE(request_invalid_argument,
                        request_status_total,
                        {{"code", "INVALID_ARGUMENT"}});
DEFINE_COUNTER_INSTANCE(request_deadline_exceeded,
                        request_status_total,
                        {{"code", "DEADLINE_EXCEEDED"}});
DEFINE_COUNTER_INSTANCE(request_resource_exhausted,
                        request_status_total,
                        {{"code", "RESOURCE_EXHAUSTED"}});
DEFINE_COUNTER_INSTANCE(request_unauthenticated,
                        request_status_total,
                        {{"code", "UNAUTHENTICATED"}});
DEFINE_COUNTER_INSTANCE(request_unavailable,
                        request_status_total,
                        {{"code", "UNAVAILABLE"}});
DEFINE_COUNTER_INSTANCE(request_unimplemented,
                        request_status_total,
                        {{"code", "UNIMPLEMENTED"}});

DEFINE_COUNTER_FAMILY(request_handling_latency_seconds,
                      "Request handling latency in seconds");
DEFINE_COUNTER_INSTANCE(chat_handling_latency_seconds,
                        request_handling_latency_seconds,
                        {{"type", "chat"}});
DEFINE_COUNTER_INSTANCE(completion_handling_latency_seconds,
                        request_handling_latency_seconds,
                        {{"type", "completion"}});

DEFINE_COUNTER(tokenization_latency_seconds,
               "Prompt tokenization latency in seconds");
DEFINE_COUNTER(chat_template_latency_seconds,
               "Chat template latency in seconds");
