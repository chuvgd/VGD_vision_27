#include <fmt/core.h>

#include <chrono>
#include <fstream>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/yaml.hpp"

const std::string keys =
  "{help h usage ? |                   | 输出命令行参数说明 }"
  "{config-path c  | configs/demo.yaml | yaml配置文件的路径}"
  "{start-index s  | 0                 | 视频起始帧下标    }"
  "{end-index e    | 0                 | 视频结束帧下标    }"
  "{bullet-speed b | 22                | 弹速(m/s)，Planner仅在10~25内有效}"
  "{@input-path    | assets/demo/demo  | avi和txt文件的路径}";

int main(int argc, char * argv[])
{
  // 读取命令行参数
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto input_path = cli.get<std::string>(0);
  auto config_path = cli.get<std::string>("config-path");
  auto start_index = cli.get<int>("start-index");
  auto end_index = cli.get<int>("end-index");
  auto bullet_speed = cli.get<double>("bullet-speed");

  tools::Plotter plotter;
  tools::Exiter exiter;

  auto video_path = fmt::format("{}.avi", input_path);
  auto text_path = fmt::format("{}.txt", input_path);
  cv::VideoCapture video(video_path);
  std::ifstream text(text_path);

  auto_aim::YOLO yolo(config_path);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Planner planner(config_path);

  // 延迟补偿参数，与Aimer中的用法一致
  auto yaml = tools::load(config_path);
  auto decision_speed = tools::read<double>(yaml, "decision_speed");
  auto high_speed_delay_time = tools::read<double>(yaml, "high_speed_delay_time");
  auto low_speed_delay_time = tools::read<double>(yaml, "low_speed_delay_time");

  cv::Mat img;
  auto t0 = std::chrono::steady_clock::now();

  video.set(cv::CAP_PROP_POS_FRAMES, start_index);
  for (int i = 0; i < start_index; i++) {
    double t, w, x, y, z;
    text >> t >> w >> x >> y >> z;
  }

  for (int frame_count = start_index; !exiter.exit(); frame_count++) {
    if (end_index > 0 && frame_count > end_index) break;

    video.read(img);
    if (img.empty()) break;

    double t, w, x, y, z;
    text >> t >> w >> x >> y >> z;
    auto timestamp = t0 + std::chrono::microseconds(int(t * 1e6));

    /// 自瞄核心逻辑

    Eigen::Quaterniond gimbal_q = {w, x, y, z};
    solver.set_R_gimbal2world(gimbal_q);

    // 视频中记录的实际云台朝向，用于开火判断中的跟随门控
    auto gimbal_yaw = tools::eulers(solver.R_gimbal2world(), 2, 1, 0)[0];

    auto yolo_start = std::chrono::steady_clock::now();
    auto armors = yolo.detect(img, frame_count);

    auto tracker_start = std::chrono::steady_clock::now();
    auto targets = tracker.track(armors, timestamp);

    auto planner_start = std::chrono::steady_clock::now();
    auto_aim::Plan plan{false};
    if (!targets.empty()) {
      // 按值拷贝，避免污染tracker内部target的状态
      auto target = targets.front();
      
      //其思路和aim.cpp中bool to_now一致，需要判断是否是真实时间，如果不是就在传入的时间戳上加延迟进行预测即可，是真实时间则需要将传入的时间戳和现在的时间戳进行作差加上延迟处理
      auto delay_time =
        std::abs(target.ekf_x()[7]) > decision_speed ? high_speed_delay_time : low_speed_delay_time;

      // detector耗时0.005s + 发弹延时
      target.predict(timestamp + std::chrono::microseconds(int((0.005 + delay_time) * 1e6)));

      // 使用带gimbal_yaw的重载，开火判断中额外加入与Shooter类似的实际云台跟随门控
      plan = planner.plan(target, bullet_speed, gimbal_yaw);
    }

    /// 调试输出

    auto finish = std::chrono::steady_clock::now();
    tools::logger()->info(
      "[{}] yolo: {:.1f}ms, tracker: {:.1f}ms, planner: {:.1f}ms", frame_count,
      tools::delta_time(tracker_start, yolo_start) * 1e3,
      tools::delta_time(planner_start, tracker_start) * 1e3,
      tools::delta_time(finish, planner_start) * 1e3);

    tools::draw_text(img, fmt::format("[{}]", tracker.state()), {10, 30}, {255, 255, 255});

    tools::draw_text(
      img,
      fmt::format(
        "is {},yaw:{:.2f},pitch:{:.2f},fire:{}", plan.control, plan.yaw * 57.3, plan.pitch * 57.3,
        plan.fire),
      {10, 60}, {0, 165, 255});

    tools::draw_text(
      img, fmt::format("gimbal yaw{:.2f}", gimbal_yaw * 57.3), {10, 90}, {255, 255, 255});

    nlohmann::json data;

    // 装甲板原始观测数据
    data["armor_num"] = armors.size();
    if (!armors.empty()) {
      const auto & armor = armors.front();
      data["armor_x"] = armor.xyz_in_world[0];
      data["armor_y"] = armor.xyz_in_world[1];
      data["armor_yaw"] = armor.ypr_in_world[0] * 57.3;
      data["armor_yaw_raw"] = armor.yaw_raw * 57.3;
      data["armor_center_x"] = armor.center_norm.x;
      data["armor_center_y"] = armor.center_norm.y;
    }

    // 云台响应情况：视频记录的实际朝向与plan的指令，角度均为度
    data["gimbal_yaw"] = gimbal_yaw * 57.3;
    data["bullet_speed"] = bullet_speed;

    data["control"] = plan.control;
    data["target_yaw"] = plan.target_yaw * 57.3;
    data["target_pitch"] = plan.target_pitch * 57.3;
    data["plan_yaw"] = plan.yaw * 57.3;
    data["plan_pitch"] = plan.pitch * 57.3;
    data["plan_yaw_vel"] = plan.yaw_vel;
    data["plan_pitch_vel"] = plan.pitch_vel;
    data["plan_yaw_acc"] = plan.yaw_acc;
    data["plan_pitch_acc"] = plan.pitch_acc;
    data["plan_yaw_err"] = tools::limit_rad(plan.yaw - gimbal_yaw) * 57.3;
    data["fire"] = plan.fire;
    data["fire_planner"] = plan.fire_planner;

    if (!targets.empty()) {
      auto target = targets.front();

      // 当前帧target更新后
      std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
      for (const Eigen::Vector4d & xyza : armor_xyza_list) {
        auto image_points =
          solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
        tools::draw_points(img, image_points, {0, 255, 0});
      }

      // planner瞄准位置
      if (plan.control) {
        Eigen::Vector4d aim_xyza = planner.debug_xyza;
        auto image_points =
          solver.reproject_armor(aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
        tools::draw_points(img, image_points, {0, 0, 255});
      }

      // 观测器内部数据
      Eigen::VectorXd x = target.ekf_x();
      data["x"] = x[0];
      data["vx"] = x[1];
      data["y"] = x[2];
      data["vy"] = x[3];
      data["z"] = x[4];
      data["vz"] = x[5];
      data["a"] = x[6] * 57.3;
      data["w"] = x[7];
      data["r"] = x[8];
      data["l"] = x[9];
      data["h"] = x[10];
      data["last_id"] = target.last_id;

      // 卡方检验数据
      data["residual_yaw"] = target.ekf().data.at("residual_yaw");
      data["residual_pitch"] = target.ekf().data.at("residual_pitch");
      data["residual_distance"] = target.ekf().data.at("residual_distance");
      data["residual_angle"] = target.ekf().data.at("residual_angle");
      data["nis"] = target.ekf().data.at("nis");
      data["nees"] = target.ekf().data.at("nees");
      data["nis_fail"] = target.ekf().data.at("nis_fail");
      data["nees_fail"] = target.ekf().data.at("nees_fail");
      data["recent_nis_failures"] = target.ekf().data.at("recent_nis_failures");
    }

    plotter.plot(data);

    cv::resize(img, img, {}, 0.5, 0.5);  // 显示时缩小图片尺寸
    cv::imshow("reprojection", img);
    auto key = cv::waitKey(30);
    if (key == 'q') break;
  }

  return 0;
}
