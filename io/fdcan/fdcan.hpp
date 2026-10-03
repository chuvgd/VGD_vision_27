#ifndef IO__FDCAN_HPP
#define IO__FDCAN_HPP

#include <Eigen/Geometry>
#include <linux/can.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>

#include "tools/thread_safe_queue.hpp"

namespace io
{
enum class FdcanMode : uint8_t
{
  IDLE = 0,
  AUTO_AIM = 1,
  SMALL_BUFF = 2,
  BIG_BUFF = 3
};

enum class FdcanShootMode : uint8_t
{
  LEFT = 0,
  RIGHT = 1,
  BOTH = 2
};

struct __attribute__((packed)) FdcanStruct
{
  uint8_t mode;  // 0: 不控制, 1: 控制云台但不开火, 2: 控制云台且开火
  float yaw;
  float yaw_vel;
  float yaw_acc;
  float pitch;
  float pitch_vel;
  float pitch_acc;
};

static_assert(sizeof(FdcanStruct) == 25);
static_assert(sizeof(FdcanStruct) <= 32);

class Fdcan
{
public:
  double bullet_speed = 0;
  FdcanMode mode = FdcanMode::IDLE;
  FdcanShootMode shoot_mode = FdcanShootMode::LEFT;
  double ft_angle = 0;

  explicit Fdcan(const std::string & config_path);
  ~Fdcan();

  Fdcan(const Fdcan &) = delete;
  Fdcan & operator=(const Fdcan &) = delete;

  Eigen::Quaterniond imu_at(std::chrono::steady_clock::time_point timestamp);

  void send(const FdcanStruct & data);
  void send(
    bool control, bool fire, float yaw, float yaw_vel, float yaw_acc, float pitch, float pitch_vel,
    float pitch_acc);

  static const char * str(FdcanMode mode);

private:
  struct IMUData
  {
    Eigen::Quaterniond q;
    std::chrono::steady_clock::time_point timestamp;
  };

  int socket_fd_ = -1;
  int send_canid_ = 0;
  int quaternion_canid_ = 0;
  int bullet_speed_canid_ = 0;

  std::atomic<bool> quit_{false};
  std::thread read_thread_;
  std::mutex tx_mutex_;

  tools::ThreadSafeQueue<IMUData> queue_{5000};
  IMUData data_ahead_;
  IMUData data_behind_;

  void open(const std::string & interface);
  void close();
  void read_thread();
  void callback(const canfd_frame & frame);
};
}  // namespace io

#endif  // IO__FDCAN_HPP
