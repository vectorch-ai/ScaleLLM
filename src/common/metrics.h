#pragma once

#include <string>

#include <prometheus/counter.h>
#include <prometheus/gauge.h>
#include <prometheus/registry.h>
#include <prometheus/text_serializer.h>

namespace llm {

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

  // get the prometheus registry
  prometheus::Registry& GetRegistry() { return registry_; }

  // get the metrics string
  std::string GetString() const {
    prometheus::TextSerializer serializer;
    return serializer.Serialize(registry_.Collect());
  }

 private:
  Metrics() = default;
  ~Metrics() = default;

  prometheus::Registry registry_;
};

// define helpful macros to hide boilerplate code
// NOLINTBEGIN(bugprone-macro-parentheses)
#define DEFINE_GAUGE(name, desc)                                \
  prometheus::Family<prometheus::Gauge>& name##_family =        \
      prometheus::BuildGauge().Name(#name).Help(desc).Register( \
          Metrics::Instance().GetRegistry());                   \
  prometheus::Gauge& name = name##_family.Add({});

#define DEFINE_COUNTER(name, desc)                                \
  prometheus::Family<prometheus::Counter>& name##_family =        \
      prometheus::BuildCounter().Name(#name).Help(desc).Register( \
          Metrics::Instance().GetRegistry());                     \
  prometheus::Counter& name = name##_family.Add({});

#define DECLARE_GAUGE(name) extern prometheus::Gauge& name;

#define DECLARE_COUNTER(name) extern prometheus::Counter& name;
// NOLINTEND(bugprone-macro-parentheses)

}  // namespace llm
