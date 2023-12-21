#include "threadpool.h"

#include <absl/synchronization/notification.h>
#include <absl/time/clock.h>
#include <gtest/gtest.h>

namespace llm {

TEST(ThreadPoolTest, ScheduleEmptyTask) {
  ThreadPool threadpool(1);
  absl::Notification notification;
  threadpool.schedule(nullptr);
}

TEST(ThreadPoolTest, ScheduleTask) {
  ThreadPool threadpool(1);
  absl::Notification notification;
  bool called = false;
  threadpool.schedule([&called, &notification]() {
    called = true;
    notification.Notify();
  });
  notification.WaitForNotification();
  EXPECT_TRUE(called);
}

TEST(ThreadPoolTest, ScheduleMultipleTasks) {
  ThreadPool threadpool(1);
  std::vector<std::string> completed_tasks;
  absl::Notification notification;

  // run frist task
  threadpool.schedule([&completed_tasks, &notification]() {
    completed_tasks.emplace_back("first");
    if (completed_tasks.size() == 2) {
      absl::SleepFor(absl::Milliseconds(100));
      notification.Notify();
    }
  });

  // run second task
  threadpool.schedule([&completed_tasks, &notification]() {
    completed_tasks.emplace_back("second");
    if (completed_tasks.size() == 2) {
      notification.Notify();
    }
  });

  notification.WaitForNotificationWithTimeout(absl::Milliseconds(200));
  EXPECT_EQ(completed_tasks.size(), 2);
  EXPECT_EQ(completed_tasks[0], "first");
  EXPECT_EQ(completed_tasks[1], "second");
}

TEST(ThreadPoolTest, MultipleThreads) {
  ThreadPool threadpool(4);
  std::atomic_uint32_t counter = 0;
  absl::Notification notification;

  for (int i = 0; i < 10; ++i) {
    threadpool.schedule([&counter, &notification]() {
      absl::SleepFor(absl::Milliseconds(100));
      counter++;
      if (counter == 10) {
        notification.Notify();
      }
    });
  }

  EXPECT_TRUE(
      notification.WaitForNotificationWithTimeout(absl::Milliseconds(400)));
  EXPECT_EQ(counter, 10);
}

}  // namespace llm
