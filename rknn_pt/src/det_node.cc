#include <stdio.h>
#include <memory>
#include <sys/time.h>
#include <ros/ros.h>
#include <std_msgs/Int32MultiArray.h>
#include <map>           // 支持std::map
#include <algorithm>     // 支持std::sort
#include <string>        // 支持std::string
#include <vector>        // 支持std::vector
#include <queue>         // 支持std::queue，用于FPS计算
#include <chrono>        // 支持时间计算，用于FPS
#include <mutex>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <actionlib/client/simple_action_client.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <move_base_msgs/MoveBaseActionResult.h>

#include "rkpt.hpp"
#include "rknnPool.hpp"
#include "rknn_pt/ObjectDetection.h" // 加入新的消息头文件

// ModelType 枚举已在 rkpt.hpp 中定义，不需要重复定义

// 物体和数字识别的类别数量定义
#define OBJ_MATERIAL_CLASS_NUM 15 // 物资类别数量
#define OBJ_DIGIT_CLASS_NUM 10    // 数字类别数量
#define NMS_THRESH 0.45           // 非极大值抑制阈值，可以调整
#define BOX_THRESH 0.65           // 物体检测框置信度阈值，可以调整

// 物资类别标签
const std::string material_labels[OBJ_MATERIAL_CLASS_NUM] = {
    "wrench", "soldering_iron", "electrodrill", 
    "tape_measure", "screwdriver", "pliers", 
    "oscilograph", "multimeter", "printer", 
    "keyboard", "mobile_phone", "mouse", 
    "headphones", "monitor", "speaker"
};

// 数字类别标签
const std::string digit_labels[OBJ_DIGIT_CLASS_NUM] = {
    "zero", "one", "two", "three", "four", 
    "five", "six", "seven", "eight", "nine"
};

// 全局变量
cv::Mat ros_frame;
std::mutex frame_mutex;
ros::Publisher det_pub;
int isInPoint = 0;  // 0表示未到达指定位置，1表示已到达指定位置
int cur_frame_id = 0;
rknnPool<RkPt, cv::Mat, DetectResultsGroup> *detectPoolObj = nullptr;
rknnPool<RkPt, cv::Mat, DetectResultsGroup> *detectPoolNum = nullptr;
bool hasObjectDetected = false;  // 用于标记是否检测到物体

// FPS计算相关变量
std::queue<double> frame_times;
double fps = 0.0;
const int MAX_FPS_QUEUE_SIZE = 30; // 用于平均FPS计算的帧数

// 定义物资类别数组
const std::vector<std::string> object_classes = {
    "wrench", "soldering_iron", "electrodrill", 
    "tape_measure", "screwdriver", "pliers", 
    "oscilograph", "multimeter", "printer", 
    "keyboard", "mobile_phone", "mouse", 
    "headphones", "monitor", "speaker"
};

// 定义数字类别数组
const std::vector<std::string> number_classes = {
    "zero", "one", "two", "three", "four", 
    "five", "six", "seven", "eight", "nine" 
};

// 数字类名与实际数字的映射
const std::map<std::string, int> digit_map = {
    {"one", 1}, {"two", 2}, {"three", 3}, {"four", 4}, {"five", 5},
    {"six", 6}, {"seven", 7}, {"eight", 8}, {"nine", 9}, {"zero", 0}
};

/**
 * 更新和计算FPS
 */
void updateFPS() {
    // 获取当前时间点
    auto current_time = std::chrono::high_resolution_clock::now();
    static auto last_time = current_time;
    
    // 计算帧间隔时间（秒）
    std::chrono::duration<double> frame_duration = current_time - last_time;
    double frame_time = frame_duration.count();
    
    // 更新时间队列
    frame_times.push(frame_time);
    if (frame_times.size() > MAX_FPS_QUEUE_SIZE) {
        frame_times.pop();
    }
    
    // 计算平均FPS（不修改队列）
    double total_time = 0.0;
    std::queue<double> temp_queue = frame_times;  // 复制队列
    while (!temp_queue.empty()) {
        total_time += temp_queue.front();
        temp_queue.pop();
    }
    
    if (total_time > 0 && frame_times.size() > 0) {
        fps = frame_times.size() / total_time;
    }
    
    // 更新上一帧时间
    last_time = current_time;
}

/**
 * 在图像上显示FPS信息
 * @param img 图像
 */
void displayFPS(cv::Mat& img) {
    if (img.empty()) return;
    
    // 格式化FPS文本
    char fps_text[50];
    sprintf(fps_text, "FPS: %.1f", fps);
    
    // 在右上角显示FPS
    int font_face = cv::FONT_HERSHEY_SIMPLEX;
    double font_scale = 0.7;
    int thickness = 2;
    int baseline = 0;
    
    cv::Size text_size = cv::getTextSize(fps_text, font_face, font_scale, thickness, &baseline);
    cv::Point text_org(img.cols - text_size.width - 10, text_size.height + 10);
    
    // 绘制文本背景
    cv::rectangle(img, 
                  cv::Point(text_org.x - 5, text_org.y - text_size.height - 5), 
                  cv::Point(text_org.x + text_size.width + 5, text_org.y + 5), 
                  cv::Scalar(0, 0, 0), -1);
    
    // 绘制FPS文本
    cv::putText(img, fps_text, text_org, font_face, font_scale, 
                cv::Scalar(0, 255, 0), thickness);
}

/**
 * 增强版显示检测结果函数，显示类别和置信度
 * @param img 图像
 * @param detections 检测结果
 */
void enhancedDrawDetections(cv::Mat& img, const std::vector<DetectionBox>& detections) {
    if (img.empty() || detections.empty()) return;
    
    for (const auto& det : detections) {
        // 检查检测框是否有效
        if (det.box.width <= 0 || det.box.height <= 0 || 
            det.box.x < 0 || det.box.y < 0 || 
            det.box.x + det.box.width > img.cols || 
            det.box.y + det.box.height > img.rows) {
            continue;
        }
        
        // 绘制检测框
        cv::rectangle(img, det.box, cv::Scalar(0, 255, 0), 2);
        
        // 准备显示文本（类别+置信度）
        char text[100];
        sprintf(text, "%s: %.2f", det.det_name.c_str(), det.score);
        
        // 设置文本参数
        int font_face = cv::FONT_HERSHEY_SIMPLEX;
        double font_scale = 0.5;
        int thickness = 1;
        int baseline = 0;
        
        // 获取文本大小
        cv::Size text_size = cv::getTextSize(text, font_face, font_scale, thickness, &baseline);
        
        // 计算文本位置（在矩形框上方）
        cv::Point text_org(det.box.x, det.box.y - 5);
        
        // 确保文本不会超出图像边界
        if (text_org.y < text_size.height) {
            text_org.y = det.box.y + det.box.height + text_size.height + 5;
        }
        if (text_org.x + text_size.width > img.cols) {
            text_org.x = img.cols - text_size.width - 5;
        }
        
        // 绘制文本背景
        cv::rectangle(img, 
                    cv::Point(text_org.x, text_org.y - text_size.height), 
                    cv::Point(text_org.x + text_size.width, text_org.y + baseline), 
                    cv::Scalar(0, 255, 0), -1);
        
        // 绘制文本
        cv::putText(img, text, text_org, font_face, font_scale, 
                    cv::Scalar(0, 0, 0), thickness);
    }
}

/**
 * 处理多位数字检测
 * 当检测到多个数字时，按从左到右顺序将它们组合成多位数
 * 
 * @param detections 检测到的数字结果
 * @param img_width 图像宽度，用于计算相对位置
 * @param img_height 图像高度，用于计算相对位置
 * @return 处理后的检测结果
 */
rknn_pt::ObjectDetection processMultiDigitNumber(const std::vector<DetectionBox>& detections, int img_width, int img_height) {
    // 如果检测数量少于2，无法组成多位数
    if (detections.size() < 2) {
        return rknn_pt::ObjectDetection(); // 返回空消息
    }

    // 复制检测结果，准备排序
    std::vector<DetectionBox> sorted_detections = detections;
    
    // 按照x坐标从左到右排序数字（遵循阅读顺序）
    std::sort(sorted_detections.begin(), sorted_detections.end(), 
        [](const DetectionBox& a, const DetectionBox& b) {
            return a.box.x < b.box.x;
        });
    
    // 检查检测结果是否真的是数字
    bool all_are_digits = true;
    for (const auto& det : sorted_detections) {
        if (digit_map.find(det.det_name) == digit_map.end()) {
            all_are_digits = false;
            break;
        }
    }
    
    if (!all_are_digits) {
        return rknn_pt::ObjectDetection(); // 不全是数字，返回空消息
    }
    
    // 计算组合后的数字值（从左到右）
    int combined_value = 0;
    std::string combined_name = "";
    
    for (const auto& det : sorted_detections) {
        auto it = digit_map.find(det.det_name);
        if (it != digit_map.end()) {
            int digit_value = it->second;
            combined_value = combined_value * 10 + digit_value;
            if (!combined_name.empty()) {
                combined_name += "_";
            }
            combined_name += det.det_name;
        }
    }
    
    // 计算所有检测框的中心点
    int total_x = 0, total_y = 0;
    float total_confidence = 0.0f;
    
    for (const auto& det : sorted_detections) {
        int center_x = det.box.x + det.box.width / 2;
        int center_y = det.box.y + det.box.height / 2;
        total_x += center_x;
        total_y += center_y;
        total_confidence += det.score;
    }
    
    // 计算平均中心点和平均置信度
    int avg_center_x = total_x / sorted_detections.size();
    int avg_center_y = total_y / sorted_detections.size();
    float avg_confidence = total_confidence / sorted_detections.size();
    
    // 创建新的检测消息
    rknn_pt::ObjectDetection result;
    result.object_type = "Combined number:" + std::to_string(combined_value);
    result.center_x = avg_center_x - img_width / 2; // 相对于图像中心的偏移
    result.center_y = avg_center_y - img_height / 2; // 相对于图像中心的偏移
    result.confidence = avg_confidence;
    
    ROS_INFO("Combined multiple digits into: %s (value: %d) at [%d, %d]", 
              result.object_type.c_str(), combined_value, result.center_x, result.center_y);
    
    return result;
}

/**
 * 在图像上绘制组合数字的可视化效果
 * 
 * @param img 要绘制的图像
 * @param detections 检测结果
 * @param combined_value 组合后的数值
 */
void drawCombinedDigits(cv::Mat& img, const std::vector<DetectionBox>& detections, int combined_value) {
    if (img.empty() || detections.empty()) return;
    
    // 按x坐标排序
    std::vector<DetectionBox> sorted_dets = detections;
    std::sort(sorted_dets.begin(), sorted_dets.end(), 
        [](const DetectionBox& a, const DetectionBox& b) {
            return a.box.x < b.box.x;
        });
    
    // 计算包围所有数字的大矩形
    int min_x = sorted_dets[0].box.x;
    int min_y = sorted_dets[0].box.y;
    int max_x = sorted_dets[0].box.x + sorted_dets[0].box.width;
    int max_y = sorted_dets[0].box.y + sorted_dets[0].box.height;
    
    for (size_t i = 1; i < sorted_dets.size(); i++) {
        min_x = std::min(min_x, sorted_dets[i].box.x);
        min_y = std::min(min_y, sorted_dets[i].box.y);
        max_x = std::max(max_x, sorted_dets[i].box.x + sorted_dets[i].box.width);
        max_y = std::max(max_y, sorted_dets[i].box.y + sorted_dets[i].box.height);
    }
    
    // 确保坐标在图像范围内
    min_x = std::max(0, min_x);
    min_y = std::max(0, min_y);
    max_x = std::min(img.cols, max_x);
    max_y = std::min(img.rows, max_y);
    
    // 检查矩形是否有效
    if (min_x >= max_x || min_y >= max_y) return;
    
    // 绘制包围矩形
    cv::Rect combined_rect(min_x, min_y, max_x - min_x, max_y - min_y);
    cv::rectangle(img, combined_rect, cv::Scalar(0, 255, 0), 2);
    
    // 在上方绘制组合数值
    std::string label = "Combined: " + std::to_string(combined_value);
    int font_face = cv::FONT_HERSHEY_SIMPLEX;
    double font_scale = 0.7;
    int thickness = 2;
    int baseline = 0;
    
    cv::Size text_size = cv::getTextSize(label, font_face, font_scale, thickness, &baseline);
    cv::Point text_org(min_x, min_y - 10);
    
    // 确保文本不会超出图像边界
    if (text_org.y < text_size.height) {
        text_org.y = max_y + text_size.height + 5;
    }
    if (text_org.x + text_size.width > img.cols) {
        text_org.x = img.cols - text_size.width - 5;
    }
    
    // 绘制文本背景
    cv::rectangle(img, 
                  cv::Point(text_org.x, text_org.y - text_size.height), 
                  cv::Point(text_org.x + text_size.width, text_org.y + baseline), 
                  cv::Scalar(0, 255, 0), -1);
    
    // 绘制文本
    cv::putText(img, label, text_org, font_face, font_scale, 
                cv::Scalar(0, 0, 0), thickness);
}

/**
 * MoveBase动作结果回调，用于更新isInPoint状态
 */
void moveBaseResultCallback(const move_base_msgs::MoveBaseActionResultConstPtr &result)
{
  if (result->status.status == actionlib_msgs::GoalStatus::SUCCEEDED) {
    ROS_INFO("导航成功到达目标点，设置isInPoint为1");
    isInPoint = 1;  // 设置为已到达
  } else {
    ROS_INFO("导航未成功到达目标点，设置isInPoint为0");
    isInPoint = 0;  // 设置为未到达
  }
}

// 图像回调函数，接收ROS图像并进行处理
void imageCallback(const sensor_msgs::ImageConstPtr &msg)
{
  // 使用局部变量而不是全局变量，减少线程同步问题
  cv::Mat local_frame;
  cv::Mat display_img;
  
  try
  {
    // 首先尝试转换ROS图像为OpenCV格式
    local_frame = cv_bridge::toCvShare(msg, "bgr8")->image;
    if (local_frame.empty()) {
      ROS_ERROR("Received empty image from camera");
      return;
    }
    
    // 使用clone创建独立副本，避免共享内存问题
    display_img = local_frame.clone();
    int width = display_img.cols;
    int height = display_img.rows;
    
    // 检查模型指针
    if (!detectPoolObj || !detectPoolNum) {
      ROS_ERROR("Detection models not initialized properly");
      return;
    }
    
    // 互斥锁保护模型操作，避免并发问题
    {
      std::lock_guard<std::mutex> lock(frame_mutex);
      
      // 更新FPS计算
      updateFPS();
      
      // 更新全局帧以供其他地方使用（如果需要）
      ros_frame = local_frame.clone();
      
      // 调用物资识别模型，使用局部帧
      detectPoolObj->put(display_img, cur_frame_id);
      
      // 获取物资识别结果
      DetectResultsGroup result_obj;
      detectPoolObj->get(result_obj);
      
      // 检查物资识别模型是否有结果
      if (!result_obj.dets.empty()) {
        // 物资识别模型有结果，处理结果
        hasObjectDetected = false;  // 先重置检测标志
        
        // 过滤置信度低于0.65的检测框
        std::vector<DetectionBox> valid_material_dets;
        for (auto &res : result_obj.dets) {
          // 只保留置信度大于等于0.65的检测框
          if (res.score >= 0.65) {
            // 检查检测框是否有效
            if (res.box.width <= 0 || res.box.height <= 0 || 
                res.box.x < 0 || res.box.y < 0 || 
                res.box.x + res.box.width > width || 
                res.box.y + res.box.height > height) {
              continue;
            }
            
            
            
            // 为物资模型结果设置正确的类别名称
            if (res.obj_id >= 0 && res.obj_id < OBJ_MATERIAL_CLASS_NUM) {
              res.det_name = material_labels[res.obj_id];
            } else {
              ROS_WARN("Invalid material ID: %d (max: %d)", res.obj_id, OBJ_MATERIAL_CLASS_NUM - 1);
              res.det_name = "unknown_material";
            }
            
            valid_material_dets.push_back(res);
            hasObjectDetected = true;  // 有有效的物体检测
          }
        }
        
        // 处理有效的物资检测结果
        for (auto &res : valid_material_dets) {
          // 计算中心坐标
          int center_x = res.box.x + res.box.width / 2;
          int center_y = res.box.y + res.box.height / 2;
          
          // 计算相对于图像中心的偏移
          int offset_center_x = center_x - width / 2;
          int offset_center_y = center_y - height / 2;
          
          // 创建检测消息
          rknn_pt::ObjectDetection det_msg;
          det_msg.object_type = res.det_name;
          det_msg.center_x = offset_center_x;
          det_msg.center_y = offset_center_y;
          det_msg.confidence = res.score;
          det_msg.is_in_position = isInPoint;  // 添加位置信息
          
          // 发布消息
          det_pub.publish(det_msg);
          
          ROS_INFO("Detected object: %s (id: %d) at [%d, %d] with confidence %.2f, isInPoint: %d", 
                   res.det_name.c_str(), res.obj_id, offset_center_x, offset_center_y, res.score, isInPoint);
        }
        
        // 绘制物资检测结果
        enhancedDrawDetections(display_img, valid_material_dets);
      } else {
        // 物资识别模型无结果
        hasObjectDetected = false;  // 标记未检测到物体
        
        // 仅当到达指定位置并且未检测到物体时才切换到数字识别模型
        if (isInPoint == 1) {
          ROS_INFO("已到达指定位置，且未检测到物体，切换到数字识别模型");
          
          detectPoolNum->put(display_img, cur_frame_id);
          
          // 获取数字识别结果
          DetectResultsGroup result_num;
          detectPoolNum->get(result_num);
          
          // 处理数字识别结果
          if (!result_num.dets.empty()) {
            // 先过滤无效的检测框和置信度低的检测框
            std::vector<DetectionBox> valid_dets;
            for (auto &res : result_num.dets) {
              // 只保留置信度大于等于0.65的检测框
              if (res.score >= 0.65) {
                
                
                // 为数字模型结果设置正确的类别名称
                if (res.obj_id >= 0 && res.obj_id < OBJ_DIGIT_CLASS_NUM) {
                  res.det_name = digit_labels[res.obj_id];
                } else {
                  ROS_WARN("Invalid digit ID: %d (max: %d)", res.obj_id, OBJ_DIGIT_CLASS_NUM - 1);
                  res.det_name = "unknown_digit";
                  continue; // 跳过无效数字
                }
                
                if (res.box.width > 0 && res.box.height > 0 && 
                    res.box.x >= 0 && res.box.y >= 0 && 
                    res.box.x + res.box.width <= width && 
                    res.box.y + res.box.height <= height) {
                  valid_dets.push_back(res);
                }
              }
            }
            
            // 首先检查是否可以组合多位数
            if (valid_dets.size() >= 2) {
              // 尝试组合多位数
              rknn_pt::ObjectDetection multi_digit_msg = processMultiDigitNumber(valid_dets, width, height);
              
              // 如果成功组合，发布组合后的结果
              if (!multi_digit_msg.object_type.empty()) {
                multi_digit_msg.is_in_position = isInPoint;  // 添加位置信息
                det_pub.publish(multi_digit_msg);
                
                // 检查是否所有检测都是数字
                bool all_are_digits = true;
                int combined_value = 0;
                std::vector<DetectionBox> digit_dets;
                
                for (const auto &res : valid_dets) {
                  // 查找数字对应的值
                  const std::string &name = res.det_name;
                  auto it = digit_map.find(name);
                  if (it != digit_map.end()) {
                    digit_dets.push_back(res);
                  } else {
                    all_are_digits = false;
                    break;
                  }
                }
                
                if (all_are_digits && digit_dets.size() >= 2) {
                  // 计算组合值
                  std::vector<DetectionBox> sorted_detections = digit_dets;
                  std::sort(sorted_detections.begin(), sorted_detections.end(), 
                      [](const DetectionBox& a, const DetectionBox& b) {
                          return a.box.x < b.box.x;
                      });
                  
                  for (const auto& det : sorted_detections) {
                    auto it = digit_map.find(det.det_name);
                    if (it != digit_map.end()) {
                        int digit_value = it->second;
                        combined_value = combined_value * 10 + digit_value;
                    }
                  }
                  
                  // 绘制组合数字
                  drawCombinedDigits(display_img, digit_dets, combined_value);
                }
              } else {
                // 如果无法组合，按单个数字处理
                enhancedDrawDetections(display_img, valid_dets);
                
                for (const auto &res : valid_dets) {
                  // 计算中心坐标
                  int center_x = res.box.x + res.box.width / 2;
                  int center_y = res.box.y + res.box.height / 2;
                  
                  // 计算相对于图像中心的偏移
                  int offset_center_x = center_x - width / 2;
                  int offset_center_y = center_y - height / 2;
                  
                  // 创建检测消息
                  rknn_pt::ObjectDetection det_msg;
                  det_msg.object_type = res.det_name;
                  det_msg.center_x = offset_center_x;
                  det_msg.center_y = offset_center_y;
                  det_msg.confidence = res.score;
                  det_msg.is_in_position = isInPoint;  // 添加位置信息
                  
                  // 发布消息
                  det_pub.publish(det_msg);
                  
                  ROS_INFO("Detected number: %s (id: %d) at [%d, %d] with confidence %.2f, isInPoint: %d", 
                           res.det_name.c_str(), res.obj_id, offset_center_x, offset_center_y, res.score, isInPoint);
                }
              }
            } else if (valid_dets.size() == 1) {
              // 单个数字处理
              enhancedDrawDetections(display_img, valid_dets);
              
              const auto &res = valid_dets[0];
              // 计算中心坐标
              int center_x = res.box.x + res.box.width / 2;
              int center_y = res.box.y + res.box.height / 2;
              
              // 计算相对于图像中心的偏移
              int offset_center_x = center_x - width / 2;
              int offset_center_y = center_y - height / 2;
              
              // 创建检测消息
              rknn_pt::ObjectDetection det_msg;
              det_msg.object_type = res.det_name;
              det_msg.center_x = offset_center_x;
              det_msg.center_y = offset_center_y;
              det_msg.confidence = res.score;
              det_msg.is_in_position = isInPoint;  // 添加位置信息
              
              // 发布消息
              det_pub.publish(det_msg);
              
              ROS_INFO("Detected number: %s (id: %d) at [%d, %d] with confidence %.2f, isInPoint: %d", 
                       res.det_name.c_str(), res.obj_id, offset_center_x, offset_center_y, res.score, isInPoint);
            } else {
              ROS_INFO("No valid digit detections");
            }
          } else {
            ROS_INFO("No numbers detected");
          }
        } else {
          ROS_INFO("未到达指定位置，继续使用物资识别模型");
        }
      }
      
      // 在右上角显示FPS
      displayFPS(display_img);
    } // 锁在这里释放
    
    // 显示处理后的图像 - 在锁之外，减少锁持有时间
    if (!display_img.empty()) {
      cv::imshow("Detection Results", display_img);
      
      // 按q键退出
      int key = cv::waitKey(1);
      if (key == 'q') {
        ros::shutdown();
      }
    }
  }
  catch (cv_bridge::Exception &e)
  {
    ROS_ERROR("Image conversion error: %s", e.what());
    return;
  }
  catch (std::exception &e)
  {
    ROS_ERROR("Exception in image callback: %s", e.what());
    return;
  }
}

int main(int argc, char **argv)
{
  // 初始化ROS节点,创建ROS节点句柄，用于与ROS系统进行交互
  ros::init(argc, argv, "det_node");
  ros::NodeHandle nh;
  
  try {
    // 创建发布器，使用自定义消息类型，话题名称为"Detect_result"
    det_pub = nh.advertise<rknn_pt::ObjectDetection>("Detect_result", 10);
    
    // 订阅MoveBaseAction结果消息
    ros::Subscriber move_base_result_sub = nh.subscribe("/move_base/result", 1, moveBaseResultCallback);
    
    // 模型路径设置
    std::string object_model_path;
    std::string number_model_path;
    
    // 置信度和NMS阈值设置
    float box_conf_threshold = BOX_THRESH;
    float nms_threshold = NMS_THRESH;
    
    // 从参数服务器获取模型路径
    nh.param<std::string>("object_model_path", object_model_path, "/home/duzhong/dzacs/src/rknn_pt/model/yolov5s_obj.rknn");
    nh.param<std::string>("number_model_path", number_model_path, "/home/duzhong/dzacs/src/rknn_pt/model/yolov5s_num.rknn");
    
    // 从参数服务器获取置信度和NMS阈值 (默认设置为0.65)
    nh.param<float>("box_conf_threshold", box_conf_threshold, 0.65);
    nh.param<float>("nms_threshold", nms_threshold, NMS_THRESH);
    
    ROS_INFO("Loading object model from: %s", object_model_path.c_str());
    ROS_INFO("Loading number model from: %s", number_model_path.c_str());
    ROS_INFO("Detection confidence threshold: %.2f, NMS threshold: %.2f", box_conf_threshold, nms_threshold);
    
    // 检查模型文件是否存在
    FILE *file_obj = fopen(object_model_path.c_str(), "r");
    if (file_obj == NULL) {
      ROS_ERROR("Cannot open object model file: %s", object_model_path.c_str());
      return -1;
    }
    fclose(file_obj);
    
    FILE *file_num = fopen(number_model_path.c_str(), "r");
    if (file_num == NULL) {
      ROS_ERROR("Cannot open number model file: %s", number_model_path.c_str());
      return -1;
    }
    fclose(file_num);
    
    // 线程池配置 - 针对RK3588S优化
    int threadNum_obj = 2; // 降低线程数，减轻NPU负担
    int threadNum_num = 2; 
    
    // 从参数服务器获取线程池配置
    nh.param<int>("threadNum_obj", threadNum_obj, 2);
    nh.param<int>("threadNum_num", threadNum_num, 2);
    
    ROS_INFO("Initializing object detection model with %d threads", threadNum_obj);
    
    // 创建并初始化模型池 - 首先只初始化物体检测模型
    detectPoolObj = new rknnPool<RkPt, cv::Mat, DetectResultsGroup>(object_model_path, threadNum_obj);
    
    if (detectPoolObj->init() != 0) {
      ROS_ERROR("Object detection model initialization failed!");
      delete detectPoolObj;//释放资源
      detectPoolObj = nullptr;
      return -1;
    }
    
    // 设置物体检测模型的置信度和NMS阈值
    RkPt* model_obj = detectPoolObj->get_model_ptr();
    if (model_obj) {
      model_obj->set_thresholds(box_conf_threshold, nms_threshold);
      model_obj->set_model_type(MODEL_MATERIAL); // 设置为物资识别模型
      ROS_INFO("Set object model thresholds: conf=%.2f, nms=%.2f", box_conf_threshold, nms_threshold);
    }
    
    ROS_INFO("Object detection model initialized successfully");
    ROS_INFO("Initializing number detection model with %d threads", threadNum_num);
    
    // 初始化数字检测模型
    detectPoolNum = new rknnPool<RkPt, cv::Mat, DetectResultsGroup>(number_model_path, threadNum_num);
    
    if (detectPoolNum->init() != 0) {
      ROS_ERROR("Number detection model initialization failed!");
      delete detectPoolNum;//释放资源
      delete detectPoolObj;//释放第一个模型资源
      detectPoolNum = nullptr;
      detectPoolObj = nullptr;
      return -1;
    }
    
    // 获取数字检测模型指针并设置属性
    RkPt* model_num = detectPoolNum->get_model_ptr();
    if (model_num) {
      model_num->set_thresholds(box_conf_threshold, nms_threshold);
      model_num->set_model_type(MODEL_DIGIT); // 设置为数字识别模型
      ROS_INFO("Set number model thresholds: conf=%.2f, nms=%.2f", box_conf_threshold, nms_threshold);
    }
    
    ROS_INFO("Number detection model initialized successfully");
    ROS_INFO("Both detection models are ready");
    
    // 安全延迟 - 等待系统稳定
    ros::Duration(1.0).sleep();
    
    // 创建图像订阅器 - 最后创建，以确保所有初始化完成
    ROS_INFO("Subscribing to camera topic: /usb_cam/image_raw");
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber image_sub = it.subscribe("/usb_cam/image_raw", 1, imageCallback);
    
    // 设置处理频率为60Hz
    ros::Rate loop_rate(60);
    
    ROS_INFO("Detection node is running. Press Ctrl+C to exit or 'q' in the image window.");
    
    while (ros::ok()) {
      cur_frame_id++; // 更新帧ID
      
      // 处理回调
      ros::spinOnce();
      
      // 控制处理频率
      loop_rate.sleep();
    }
  }
  catch (std::exception &e) {
    ROS_ERROR("Exception in main: %s", e.what());
  }
  catch (...) {
    ROS_ERROR("Unknown exception in main");
  }
  
  // 安全释放资源
  ROS_INFO("Shutting down and cleaning up resources...");
  
  if (detectPoolObj) {
    delete detectPoolObj;
    detectPoolObj = nullptr;
  }
  
  if (detectPoolNum) {
    delete detectPoolNum;
    detectPoolNum = nullptr;
  }
  
  // 关闭所有OpenCV窗口
  cv::destroyAllWindows();
  
  ROS_INFO("Cleanup complete, exiting.");
  return 0;
}