#include <fmt/core.h>

#include <chrono>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <optional>

#include "io/camera.hpp"
#include "io/fdcan/fdcan.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"

using namespace std::chrono_literals;

const std::string keys =
  "{help h usage ? |                        | 输出命令行参数说明}"
  "{@config-path   | configs/hero.yaml | 位置参数，yaml配置文件路径 }";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  tools::Exiter exiter;
  tools::Plotter plotter;

  io::Fdcan fdcan(config_path);
  io::Camera camera(config_path);

  auto_aim::YOLO yolo(config_path, false);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Planner planner(config_path);

  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;
  auto last_mode = io::FdcanMode::IDLE;

  while (!exiter.exit()) {
    camera.read(img, timestamp);
    auto q = fdcan.imu_at(timestamp - 1ms);

    solver.set_R_gimbal2world(q);
    auto gimbal_pos = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);

    if (last_mode != fdcan.mode) {
      tools::logger()->info("Switch to {}", io::Fdcan::str(fdcan.mode));
      last_mode = fdcan.mode;
    }

    std::optional<auto_aim::Target> target;
    auto_aim::Plan plan{false};

    if (fdcan.mode == io::FdcanMode::AUTO_AIM) {
      auto armors = yolo.detect(img);
      auto targets = tracker.track(armors, timestamp);

      if (!targets.empty()) target = targets.front();

      // Planner 负责六维轨迹；最后一个门控使用真实云台位置判断是否允许开火。
      plan = planner.plan(target, fdcan.bullet_speed, gimbal_pos[0]);
    }

    fdcan.send(
      plan.control, plan.fire, plan.yaw, plan.yaw_vel, plan.yaw_acc, plan.pitch, plan.pitch_vel,
      plan.pitch_acc);

    /// debug
    tools::draw_text(
      img, fmt::format("[{}]", io::Fdcan::str(fdcan.mode)), {10, 30}, {255, 255, 255});
    tools::draw_text(img, fmt::format("[{}]", tracker.state()), {10, 60}, {255, 255, 255});

    nlohmann::json data;
    data["gimbal_yaw"] = gimbal_pos[0] * 57.3;
    data["gimbal_pitch"] = -gimbal_pos[1] * 57.3;
    data["bullet_speed"] = fdcan.bullet_speed;
    data["control"] = plan.control ? 1 : 0;
    data["fire_planner"] = plan.fire_planner ? 1 : 0;
    data["fire"] = plan.fire ? 1 : 0;
    data["gimbal_yaw_error"] =
      std::abs(tools::limit_rad(gimbal_pos[0] - plan.yaw)) * 57.3;
    data["cmd_yaw"] = plan.yaw * 57.3;
    data["cmd_yaw_vel"] = plan.yaw_vel;
    data["cmd_yaw_acc"] = plan.yaw_acc;
    data["cmd_pitch"] = plan.pitch * 57.3;
    data["cmd_pitch_vel"] = plan.pitch_vel;
    data["cmd_pitch_acc"] = plan.pitch_acc;

    if (target.has_value()) {
      auto armor_xyza_list = target->armor_xyza_list();
      for (const Eigen::Vector4d & xyza : armor_xyza_list) {
        auto image_points =
          solver.reproject_armor(xyza.head(3), xyza[3], target->armor_type, target->name);
        tools::draw_points(img, image_points, {0, 255, 0});
      }

      auto image_points = solver.reproject_armor(
        planner.debug_xyza.head(3), planner.debug_xyza[3], target->armor_type, target->name);
      tools::draw_points(img, image_points, {0, 0, 255});

      Eigen::VectorXd x = target->ekf_x();
      data["target_x"] = x[0];
      data["target_vx"] = x[1];
      data["target_y"] = x[2];
      data["target_vy"] = x[3];
      data["target_z"] = x[4];
      data["target_vz"] = x[5];
      data["target_a"] = x[6] * 57.3;
      data["target_w"] = x[7];
      data["target_r"] = x[8];
      data["target_l"] = x[9];
      data["target_h"] = x[10];
      data["target_id"] = target->last_id;
    }

    plotter.plot(data);

    cv::resize(img, img, {}, 0.5, 0.5);
    cv::imshow("hero", img);
    if (cv::waitKey(1) == 'q') break;
  }

  // 退出前持续失能，避免云台保留最后一次 MPC 控制。
  fdcan.send(false, false, 0, 0, 0, 0, 0, 0);

  return 0;
}
