#include "fdcan.hpp"

#include <linux/can.h>
#include <linux/can/raw.h>
#include <net/if.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <system_error>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"

namespace io
{
Fdcan::Fdcan(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  auto interface = tools::read<std::string>(yaml, "can_interface");
  send_canid_ = tools::read<int>(yaml, "send_canid");
  quaternion_canid_ = tools::read<int>(yaml, "quaternion_canid");
  bullet_speed_canid_ = tools::read<int>(yaml, "bullet_speed_canid");

  open(interface);
  read_thread_ = std::thread(&Fdcan::read_thread, this);

  tools::logger()->info("[Fdcan] Waiting for q...");
  queue_.pop(data_ahead_);
  queue_.pop(data_behind_);
  tools::logger()->info("[Fdcan] Opened.");
}

Fdcan::~Fdcan()
{
  quit_ = true;
  if (read_thread_.joinable()) read_thread_.join();
  close();
}

Eigen::Quaterniond Fdcan::imu_at(std::chrono::steady_clock::time_point timestamp)
{
  if (data_behind_.timestamp < timestamp) data_ahead_ = data_behind_;

  while (true) {
    queue_.pop(data_behind_);
    if (data_behind_.timestamp > timestamp) break;
    data_ahead_ = data_behind_;
  }

  Eigen::Quaterniond q_a = data_ahead_.q.normalized();
  Eigen::Quaterniond q_b = data_behind_.q.normalized();
  auto t_a = data_ahead_.timestamp;
  auto t_b = data_behind_.timestamp;
  auto t_ab = tools::delta_time(t_a, t_b);
  auto t_ac = tools::delta_time(t_a, timestamp);

  auto k = t_ac / t_ab;
  return q_a.slerp(k, q_b).normalized();
}

void Fdcan::send(const FdcanStruct & data)
{
  if (socket_fd_ < 0) return;

  std::lock_guard<std::mutex> lock(tx_mutex_);

  canfd_frame frame{};
  frame.can_id = send_canid_;
  // CAN FD 硬件使用离散 DLC，25 字节结构补零后按 32 字节发送。
  frame.len = 32;
  std::memcpy(frame.data, &data, 32);

  try {
    auto written = ::write(socket_fd_, &frame, CANFD_MTU);
    if (written != CANFD_MTU)
      throw std::system_error(errno, std::generic_category(), "write(canfd_frame)");
  } catch (const std::exception & e) {
    tools::logger()->warn("[Fdcan] Failed to send: {}", e.what());
  }
}

void Fdcan::send(
  bool control, bool fire, float yaw, float yaw_vel, float yaw_acc, float pitch, float pitch_vel,
  float pitch_acc)
{
  FdcanStruct data{};
  data.mode = control ? static_cast<uint8_t>(fire ? 2 : 1) : 0;
  data.yaw = yaw;
  data.yaw_vel = yaw_vel;
  data.yaw_acc = yaw_acc;
  data.pitch = pitch;
  data.pitch_vel = pitch_vel;
  data.pitch_acc = pitch_acc;

  send(data);
}

const char * Fdcan::str(FdcanMode mode)
{
  switch (mode) {
    case FdcanMode::IDLE:
      return "IDLE";
    case FdcanMode::AUTO_AIM:
      return "AUTO_AIM";
    case FdcanMode::SMALL_BUFF:
      return "SMALL_BUFF";
    case FdcanMode::BIG_BUFF:
      return "BIG_BUFF";
    default:
      return "INVALID";
  }
}

void Fdcan::open(const std::string & interface)
{
  socket_fd_ = socket(PF_CAN, SOCK_RAW, CAN_RAW);
  if (socket_fd_ < 0) throw std::system_error(errno, std::generic_category(), "socket(PF_CAN)");

  int enable_fd = 1;
  if (setsockopt(socket_fd_, SOL_CAN_RAW, CAN_RAW_FD_FRAMES, &enable_fd, sizeof(enable_fd)) < 0) {
    auto error = errno;
    close();
    throw std::system_error(error, std::generic_category(), "setsockopt(CAN_RAW_FD_FRAMES)");
  }

  ifreq ifr{};
  std::strncpy(ifr.ifr_name, interface.c_str(), IFNAMSIZ - 1);

  if (ioctl(socket_fd_, SIOCGIFINDEX, &ifr) < 0) {
    auto error = errno;
    close();
    throw std::system_error(error, std::generic_category(), "ioctl(SIOCGIFINDEX)");
  }

  sockaddr_can addr{};
  addr.can_family = AF_CAN;
  addr.can_ifindex = ifr.ifr_ifindex;

  if (bind(socket_fd_, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
    auto error = errno;
    close();
    throw std::system_error(error, std::generic_category(), "bind(PF_CAN)");
  }

  tools::logger()->info(
    "[Fdcan] Opened {}, send CAN ID: 0x{:X}, payload size: {}", interface, send_canid_,
    32);
}

void Fdcan::close()
{
  if (socket_fd_ >= 0) {
    ::close(socket_fd_);
    socket_fd_ = -1;
  }
}

void Fdcan::read_thread()
{
  while (!quit_) {
    pollfd pfd{socket_fd_, POLLIN, 0};
    auto ready = poll(&pfd, 1, 100);

    if (ready < 0) {
      if (errno == EINTR) continue;
      tools::logger()->warn("[Fdcan] poll failed: {}", std::strerror(errno));
      break;
    }

    if (ready == 0) continue;

    canfd_frame frame{};
    auto bytes = recv(socket_fd_, &frame, CANFD_MTU, 0);
    if (bytes < 0) {
      if (errno == EINTR) continue;
      tools::logger()->warn("[Fdcan] recv failed: {}", std::strerror(errno));
      continue;
    }

    callback(frame);
  }
}

void Fdcan::callback(const canfd_frame & frame)
{
  auto timestamp = std::chrono::steady_clock::now();

  if (frame.can_id == quaternion_canid_) {
    if (frame.len < 8) return;

    auto x = static_cast<int16_t>(frame.data[0] << 8 | frame.data[1]) / 1e4;
    auto y = static_cast<int16_t>(frame.data[2] << 8 | frame.data[3]) / 1e4;
    auto z = static_cast<int16_t>(frame.data[4] << 8 | frame.data[5]) / 1e4;
    auto w = static_cast<int16_t>(frame.data[6] << 8 | frame.data[7]) / 1e4;

    if (std::abs(x * x + y * y + z * z + w * w - 1) > 1e-2) {
      tools::logger()->warn("[Fdcan] Invalid q: {} {} {} {}", w, x, y, z);
      return;
    }

    queue_.push({{w, x, y, z}, timestamp});
  } else if (frame.can_id == bullet_speed_canid_) {
    if (frame.len < 6) return;

    bullet_speed = static_cast<int16_t>(frame.data[0] << 8 | frame.data[1]) / 1e2;
    mode = static_cast<FdcanMode>(frame.data[2]);
    shoot_mode = static_cast<FdcanShootMode>(frame.data[3]);
    ft_angle = static_cast<int16_t>(frame.data[4] << 8 | frame.data[5]) / 1e4;

    static auto last_log_time = std::chrono::steady_clock::time_point::min();
    auto now = std::chrono::steady_clock::now();

    if (bullet_speed > 0 && tools::delta_time(now, last_log_time) >= 1.0) {
      tools::logger()->info(
        "[Fdcan] Bullet speed: {:.2f} m/s, Mode: {}, FT angle: {:.2f} rad", bullet_speed,
        Fdcan::str(mode), ft_angle);
      last_log_time = now;
    }
  }
}
}  // namespace io
