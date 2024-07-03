#pragma once

#include <prometheus/counter.h>
#include <prometheus/family.h>
#include <prometheus/gauge.h>
#include <prometheus/histogram.h>
#include <prometheus/labels.h>
#include <prometheus/registry.h>
#include <prometheus/text_serializer.h>

#include <string>

#include "macros.h"
#include "timer.h"

namespace llm {

using prometheus::Counter;
using prometheus::Gauge;
using prometheus::Histogram;
using prometheus::Info;
using prometheus::Summary;

class Metrics final {
 public:
  Metrics(const Metrics&) = delete;
  Metrics& operator=(const Metrics&) = delete;

  Metrics(Metrics&&) = delete;
  Metrics& operator=(Metrics&&) = delete;

  // a sington class
  static Metrics& Instance() {
    static Metrics instance;
    return instance;
  }

  // get the metrics string
  std::string GetString() const {
    prometheus::TextSerializer serializer;
    return serializer.Serialize(registry_.Collect());
  }

  // helper functions to define metrics
  prometheus::Family<prometheus::Gauge>& BuildGauge(const std::string& name,
                                                    const std::string& desc) {
    return prometheus::BuildGauge().Name(name).Help(desc).Register(registry_);
  }

  prometheus::Family<prometheus::Counter>& BuildCounter(
      const std::string& name,
      const std::string& desc) {
    return prometheus::BuildCounter().Name(name).Help(desc).Register(registry_);
  }

  prometheus::Family<prometheus::Histogram>& BuildHistogram(
      const std::string& name,
      const std::string& desc) {
    return prometheus::BuildHistogram().Name(name).Help(desc).Register(
        registry_);
  }

 private:
  Metrics() = default;
  ~Metrics() = default;

  prometheus::Registry registry_;
};

class AutoCounter final {
 public:
  AutoCounter(prometheus::Counter& counter) : counter_(counter) {}

  ~AutoCounter() {
    // increment the counter
    counter_.Increment(timer_.elapsed_seconds());
  }

 private:
  // NOLINTNEXTLINE
  prometheus::Counter& counter_;

  // the timer
  Timer timer_;
};

}  // namespace llm

// define helpful macros to hide boilerplate code
// NOLINTBEGIN(bugprone-macro-parentheses)
// extern const prometheus::Histogram::BucketBoundaries default_buckets;

// define gauge
// a gauge is a metric that represents a single numerical value that can
// arbitrarily go up and down.
#define DEFINE_GAUGE(name, desc)    \
  prometheus::Gauge& GAUGE_##name = \
      llm::Metrics::Instance().BuildGauge(#name, desc).Add({});

#define DEFINE_GAUGE_FAMILY(name, desc)                  \
  prometheus::Family<prometheus::Gauge>& name##_family = \
      llm::Metrics::Instance().BuildGauge(#name, desc);

#define DEFINE_GAUGE_INSTANCE(alias, name, ...) \
  prometheus::Gauge& GAUGE_##alias = name##_family.Add(__VA_ARGS__);

#define GAUGE_SET(name, value) GAUGE_##name.Set(value);

#define GAUGE_ADD(name, value) GAUGE_##name.Increment(value);

#define GAUGE_INC(name) GAUGE_##name.Increment();

// define counter
// a counter is a monotonically increasing counter whose value can only increase
// or be reset to zero on restart.
#define DEFINE_COUNTER(name, desc)      \
  prometheus::Counter& COUNTER_##name = \
      llm::Metrics::Instance().BuildCounter(#name, desc).Add({});

#define DEFINE_COUNTER_FAMILY(name, desc)                  \
  prometheus::Family<prometheus::Counter>& name##_family = \
      llm::Metrics::Instance().BuildCounter(#name, desc);

#define DEFINE_COUNTER_INSTANCE(alias, name, ...) \
  prometheus::Counter& COUNTER_##alias = name##_family.Add(__VA_ARGS__);

#define COUNTER_ADD(name, value) COUNTER_##name.Increment(value);

#define COUNTER_INC(name) COUNTER_##name.Increment();

// Declares a latency counter having a variable name based on line number.
// example: AUTO_COUNTER(a_counter_name);
#define AUTO_COUNTER(name) llm::AutoCounter LLM_ANON_VAR(name)(COUNTER_##name);

// define histogram
// a histogram samples observations (usually things like request durations or
// response sizes) and counts them in configurable buckets. It also provides a
// sum of all observed values.
#define DEFINE_HISTOGRAM(name, desc, ...)                                    \
  prometheus::Histogram& HISTOGRAM_##name = llm::Metrics::Instance()         \
                                                .BuildHistogram(#name, desc) \
                                                .Add({}, __VA_ARGS__);

#define DEFINE_HISTOGRAM_FAMILY(name, desc)                  \
  prometheus::Family<prometheus::Histogram>& name##_family = \
      llm::Metrics::Instance().BuildHistogram(#name, desc);

#define DEFINE_HISTOGRAM_INSTANCE(alias, name, ...) \
  prometheus::Histogram& HISTOGRAM_##alias = name##_family.Add(__VA_ARGS__);

#define HISTOGRAM_OBSERVE(name, value) HISTOGRAM_##name.Observe(value);

// declare gauge
#define DECLARE_GAUGE(name) extern prometheus::Gauge& GAUGE_##name;

#define DECLARE_GAUGE_INSTANCE(alias) extern prometheus::Gauge& GAUGE_##alias;

#define DECLARE_GAUGE_FAMILY(name) \
  extern prometheus::Family<prometheus::Gauge>& name##_family;

// declare counter
#define DECLARE_COUNTER(name) extern prometheus::Counter& COUNTER_##name;

#define DECLARE_COUNTER_INSTANCE(alias) \
  extern prometheus::Counter& COUNTER_##alias;

#define DECLARE_COUNTER_FAMILY(name) \
  extern prometheus::Family<prometheus::Counter>& name##_family;

// declare histogram
#define DECLARE_HISTOGRAM(name) extern prometheus::Histogram& HISTOGRAM_##name;

#define DECLARE_HISTOGRAM_INSTANCE(alias) \
  extern prometheus::Histogram& HISTOGRAM_##alias;

#define DECLARE_HISTOGRAM_FAMILY(name) \
  extern prometheus::Family<prometheus::Histogram>& name##_family;
// NOLINTEND(bugprone-macro-parentheses)
