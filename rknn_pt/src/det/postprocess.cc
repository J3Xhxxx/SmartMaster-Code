#include "det/postprocess.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <iostream>

#include <set>
#include <vector>

// 物资类别标签数组（0-14）  
static const char *material_labels[OBJ_MATERIAL_CLASS_NUM] = {
    "wrench", "soldering_iron", "electrodrill", 
    "tape_measure", "screwdriver", "pliers", 
    "oscilograph", "multimeter", "printer", 
    "keyboard", "mobile_phone", "mouse", 
    "headphones", "monitor", "speaker"
};

// 数字类别标签数组（15-24）
static const char *digit_labels[OBJ_DIGIT_CLASS_NUM] = {
    "one", "two", "three", "four", "five", 
    "six", "seven", "eight", "nine", "zero"
};

// 合并的类别标签数组（用于实际处理）
static const char *labels[OBJ_CLASS_NUM] = {
    "wrench", "soldering_iron", "electrodrill", 
    "tape_measure", "screwdriver", "pliers", 
    "oscilograph", "multimeter", "printer", 
    "keyboard", "mobile_phone", "mouse", 
    "headphones", "monitor", "speaker",
    "one", "two", "three", "four", "five", 
    "six", "seven", "eight", "nine", "zero"
};

// YOLO锚点框配置 三个检测层各自的锚点尺寸(无需修改)
// 格式说明：每个数组包含两个锚点的宽高参数(w,h,w,h,w,h)
const int anchor0[6] = {10, 13, 16, 30, 33, 23};  // 浅层特征锚点（小目标）
const int anchor1[6] = {30, 61, 62, 45, 59, 119}; // 中层特征锚点  
const int anchor2[6] = {116, 90, 156, 198, 373, 326}; // 深层特征锚点（大目标）

// 可视化相关配置
static const int cnum = 2;              // 颜色池大小
static cv::Scalar_<int> randColor[cnum];// 随机颜色数组（用于不同类别渲染）
static bool init_colors = false;        // 颜色初始化标记

/* 数值截断函数（INT8量化用，无需修改） */
inline static int clamp(float val, int min, int max) { 
    return val > min ? (val < max ? val : max) : min; 
}


/////////////////////////////////////////这里是yolo的内容///////////////////////////////////



/* 计算两个矩形框的交并比(IoU) （无需修改）*/
static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0,
                               float xmin1, float ymin1, float xmax1, float ymax1) {
    // 计算交集区域
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
    float i = w * h;
    
    // 计算并集区域
    float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) 
            + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
    
    return u <= 0.f ? 0.f : (i / u);
}

/* 非极大值抑制(NMS)实现 （无需修改）*/
static int nms(int validCount, std::vector<float> &outputLocations,
               std::vector<int> classIds, std::vector<int> &order,
               int filterId, float threshold) {
    for (int i = 0; i < validCount; ++i) {
        // 跳过已抑制或非目标类别的检测框
        if (order[i] == -1 || classIds[i] != filterId) continue;
        
        int n = order[i];  // 当前基准框索引
        
        // 与后续检测框比较
        for (int j = i + 1; j < validCount; ++j) {
            int m = order[j];
            // 跳过已抑制或非目标类别的比较框
            if (m == -1 || classIds[j] != filterId) continue;
            
            // 解算两个框的坐标参数
            float xmin0 = outputLocations[n * 4 + 0];
            float ymin0 = outputLocations[n * 4 + 1];
            float xmax0 = xmin0 + outputLocations[n * 4 + 2]; // xmin + width
            float ymax0 = ymin0 + outputLocations[n * 4 + 3]; // ymin + height
            
            float xmin1 = outputLocations[m * 4 + 0];
            float ymin1 = outputLocations[m * 4 + 1];
            float xmax1 = xmin1 + outputLocations[m * 4 + 2];
            float ymax1 = ymin1 + outputLocations[m * 4 + 3];
            
            // 计算IoU并执行抑制
            float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0,
                                        xmin1, ymin1, xmax1, ymax1);
            if (iou > threshold) {
                order[j] = -1; // 标记重叠框为抑制状态
            }
        }
    }
    return 0;
}

// 函数 quick_sort_indice_inverse 实现了基于快速排序算法的反向索引排序
static int quick_sort_indice_inverse(std::vector<float> &input, int left, int right, std::vector<int> &indices)
{
  float key; // 当前基准值
  int key_index; // 当前基准值在 indices 中的索引
  int low = left; // 指向当前分区的左边界
  int high = right; // 指向当前分区的右边界

  // 检查左边界是否小于右边界，确保有待排序的元素
  if (left < right)
  {
    key_index = indices[left]; // 记录基准值索引
    key = input[left]; // 记录基准值

    // 开始进行分区操作
    while (low < high)
    {
      // 从右侧找到第一个小于等于基准值的元素
      while (low < high && input[high] <= key)
      {
        high--; // 右指针向左移动
      }
      // 将高指针指向的元素移动到低指针的位置
      input[low] = input[high];
      indices[low] = indices[high]; // 同样移动索引

      // 从左侧找到第一个大于等于基准值的元素
      while (low < high && input[low] >= key)
      {
        low++; // 低指针向右移动
      }
      // 将低指针指向的元素移动到高指针的位置
      input[high] = input[low];
      indices[high] = indices[low]; // 同样移动索引
    }

    // 将基准值放回当前分区的正确位置
    input[low] = key;
    indices[low] = key_index; // 同样更新索引

    // 递归调用对左侧子数组进行排序
    quick_sort_indice_inverse(input, left, low - 1, indices);
    // 递归调用对右侧子数组进行排序
    quick_sort_indice_inverse(input, low + 1, right, indices);
  }
  // 返回基准值的最终位置
  return low;
}

// 将浮点数剪切到指定的范围内
inline static int32_t __clip(float val, float min, float max)
{
  float f = val <= min ? min : (val >= max ? max : val); // 如果val小于min则返回min，如果大于max则返回max，否则返回val
  return f; // 返回剪切后的值
}

// 将浮点数转换为量化后的8位整数（INT8量化）（使用affine量化方法）
static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
{
  float dst_val = (f32 / scale) + zp; // 使用affine量化公式将浮点数转换为量化值
  int8_t res = (int8_t)__clip(dst_val, -128, 127); // 将量化值剪切到8位整数的范围内
  return res; // 返回量化后的8位整数值
}

// 将量化后的8位整数转换回浮点数（INT8反量化）（使用affine反量化方法）
static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) 
{ 
  return ((float)qnt - (float)zp) * scale; // 使用affine反量化公式将8位整数转换回浮点数
}

// 处理输入数据，提取目标检测框信息
// 输入：输入数据，锚点数组，网格高度，网格宽度，模型输入高度，模型输入宽度，步长，结果数组，类别数组，置信度阈值，量化参数
static int process(int8_t *input, int *anchor, int grid_h, int grid_w, int height, int width, int stride,
                   std::vector<float> &boxes, std::vector<float> &objProbs, std::vector<int> &classId, float threshold,
                   int32_t zp, float scale)
{
  int validCount = 0; // 有效目标计数
  int grid_len = grid_h * grid_w; // 网格长度（网格总数）
  int8_t thres_i8 = qnt_f32_to_affine(threshold, zp, scale); // 将阈值转换为量化值

  // 遍历每个anchor
  for (int a = 0; a < 3; a++)
  {
    // 遍历每个网格行
    for (int i = 0; i < grid_h; i++)
    {
      // 遍历每个网格列
      for (int j = 0; j < grid_w; j++)
      {
        // 获取当前网格单元的box置信度
        int8_t box_confidence = input[(PROP_BOX_SIZE * a + 4) * grid_len + i * grid_w + j];
        // 如果box置信度大于阈值，则进一步处理
        if (box_confidence >= thres_i8)
        {
          // 计算当前网格单元的偏移量
          int offset = (PROP_BOX_SIZE * a) * grid_len + i * grid_w + j;
          int8_t *in_ptr = input + offset; // 指向当前网格单元的指针

          // 反量化获取box的中心x坐标，并调整为实际坐标
          float box_x = (deqnt_affine_to_f32(*in_ptr, zp, scale)) * 2.0 - 0.5;
          // 反量化获取box的中心y坐标，并调整为实际坐标
          float box_y = (deqnt_affine_to_f32(in_ptr[grid_len], zp, scale)) * 2.0 - 0.5;
          // 反量化获取box的宽度，并调整为实际宽度
          float box_w = (deqnt_affine_to_f32(in_ptr[2 * grid_len], zp, scale)) * 2.0;
          // 反量化获取box的高度，并调整为实际高度
          float box_h = (deqnt_affine_to_f32(in_ptr[3 * grid_len], zp, scale)) * 2.0;

          // 根据网格坐标和stride调整box的中心坐标
          box_x = (box_x + j) * (float)stride;
          box_y = (box_y + i) * (float)stride;
          // 使用anchor调整box的宽度和高度
          box_w = box_w * box_w * (float)anchor[a * 2];
          box_h = box_h * box_h * (float)anchor[a * 2 + 1];
          // 调整box的中心坐标到box的左上角坐标
          box_x -= (box_w / 2.0);
          box_y -= (box_h / 2.0);

          // 获取当前网格单元的最大类概率
          int8_t maxClassProbs = in_ptr[5 * grid_len];
          int maxClassId = 0; // 初始化最大类ID为0
          // 遍历每个类的概率，找到最大类概率及其对应的类ID
          for (int k = 1; k < OBJ_CLASS_NUM; ++k)
          {
            int8_t prob = in_ptr[(5 + k) * grid_len];
            if (prob > maxClassProbs) // 如果当前类的概率大于最大类概率
            {
              maxClassId = k; // 更新最大类ID
              maxClassProbs = prob; // 更新最大类概率
            }
          }
          // 如果最大类概率大于阈值，则将box信息添加到结果中
          if (maxClassProbs > thres_i8)
          {
            // 反量化box置信度和最大类概率，并将两者相乘后添加到objProbs向量中
            objProbs.push_back((deqnt_affine_to_f32(maxClassProbs, zp, scale)) * (deqnt_affine_to_f32(box_confidence, zp, scale)));
            // 将最大类ID添加到classId向量中
            classId.push_back(maxClassId);
            // 增加有效目标计数
            validCount++;
            // 将box的左上角坐标、宽度和高度添加到boxes向量中
            boxes.push_back(box_x);
            boxes.push_back(box_y);
            boxes.push_back(box_w);
            boxes.push_back(box_h);
          }
        }
      }
    }
  }
  return validCount; // 返回有效目标的数量
}

// 后处理函数，对输入的三个检测层的输出进行处理，提取目标检测框信息，并进行非极大值抑制      实际上也就是他这里的yolo的检测方法
// 输入：三个检测层的输出，模型输入尺寸，置信度阈值，非极大值抑制阈值，量化参数，检测结果组
int post_process(int8_t *input0, int8_t *input1, int8_t *input2, int model_in_h, int model_in_w, float conf_threshold,
                 float nms_threshold, BOX_RECT pads, float scale_w, float scale_h, std::vector<int32_t> &qnt_zps,
                 std::vector<float> &qnt_scales, DetectResultsGroup *group)
{
  memset(group, 0, sizeof(DetectResultsGroup));

  std::vector<float> filterBoxes;
  std::vector<float> objProbs;
  std::vector<int> classId;

  // stride 8
  int stride0 = 8;
  int grid_h0 = model_in_h / stride0;
  int grid_w0 = model_in_w / stride0;
  int validCount0 = 0;
  validCount0 = process(input0, (int *)anchor0, grid_h0, grid_w0, model_in_h, model_in_w, stride0, filterBoxes, objProbs,
                        classId, conf_threshold, qnt_zps[0], qnt_scales[0]);

  // stride 16
  int stride1 = 16;
  int grid_h1 = model_in_h / stride1;
  int grid_w1 = model_in_w / stride1;
  int validCount1 = 0;
  validCount1 = process(input1, (int *)anchor1, grid_h1, grid_w1, model_in_h, model_in_w, stride1, filterBoxes, objProbs,
                        classId, conf_threshold, qnt_zps[1], qnt_scales[1]);

  // stride 32
  int stride2 = 32;
  int grid_h2 = model_in_h / stride2;
  int grid_w2 = model_in_w / stride2;
  int validCount2 = 0;
  validCount2 = process(input2, (int *)anchor2, grid_h2, grid_w2, model_in_h, model_in_w, stride2, filterBoxes, objProbs,
                        classId, conf_threshold, qnt_zps[2], qnt_scales[2]);

  int validCount = validCount0 + validCount1 + validCount2;
  // no object detect
  if (validCount <= 0)
  {
    return 0;
  }

  std::vector<int> indexArray;
  for (int i = 0; i < validCount; ++i)
  {
    indexArray.push_back(i);
  }

  quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

  std::set<int> class_set(std::begin(classId), std::end(classId));

  for (auto c : class_set)
  {
    nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
  }

  /* box valid detect target */
  for (int i = 0; i < validCount; ++i)
  {
    // if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE)
    if (indexArray[i] == -1)
    {
      continue;
    }
    int n = indexArray[i];

    float x1 = filterBoxes[n * 4 + 0] - pads.left;
    float y1 = filterBoxes[n * 4 + 1] - pads.top;
    float x2 = x1 + filterBoxes[n * 4 + 2];
    float y2 = y1 + filterBoxes[n * 4 + 3];
    int id = classId[n];
    float obj_conf = objProbs[i];

    DetectionBox new_box;
    int _x1 = (int)(clamp(x1, 0, model_in_w) / scale_w);
    int _y1 = (int)(clamp(y1, 0, model_in_h) / scale_h);
    int _x2 = (int)(clamp(x2, 0, model_in_w) / scale_w);
    int _y2 = (int)(clamp(y2, 0, model_in_h) / scale_h);
    new_box.box = cv::Rect_<int>(_x1, _y1, _x2 - _x1, _y2 - _y1);
    new_box.score = objProbs[i];
    new_box.obj_id = id;  // 保存原始的ID
    
    // 模型类型会在RkPt::infer函数中设置
    // 默认设置为物资识别模型类型
    new_box.model_type = MODEL_MATERIAL;
    
    // 初始化标签为"unknown"
    new_box.det_name = "unknown";
    
    // 这里只根据ID设置标签名称，具体的处理逻辑在imageCallback中通过model_type确定
    // 因为不同模型使用相同的ID可能表示不同类别
    if (id >= 0) {
      if (id < OBJ_MATERIAL_CLASS_NUM) {
        // 可能是物资模型里的ID
        new_box.det_name = material_labels[id];
        printf("Detected potential material object: %s (id: %d), score: %.2f\n", 
               new_box.det_name.c_str(), id, new_box.score);
      }
      
      if (id < OBJ_DIGIT_CLASS_NUM) {
        // 也可能是数字模型里的ID
        // 记录可能的数字类别名称（最终要根据model_type来决定使用哪个）
        printf("Detected potential digit: %s (id: %d), score: %.2f\n", 
               digit_labels[id], id, new_box.score);
      }
    } else {
      printf("Warning: Invalid class ID: %d\n", id);
    }
    
    group->dets.push_back(new_box);
  }

  return 0;
}




/////////////////////////////////////////yolo相关部分在这里结束////////////////////////////////////////





//绘制矩形框
int draw_image_detect(cv::Mat &cur_img, std::vector<DetectionBox> &results, int cur_frame_id)
{
  char text[256];
  for (const auto& res : results)
  {
    sprintf(text, "%s", res.det_name.c_str());//将res中的名字转化为字符串放到text中
    cv::rectangle(cur_img, res.box, cv::Scalar(256, 0, 0, 256), 3);//在res.box确定的位置画矩形框
    cv::putText(cur_img, text, cv::Point(res.box.x, res.box.y + 12), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
                                                    //在res.box的左上角画名字
  }
  std::ostringstream oss;

  oss << "detect_" << std::setfill('0') << std::setw(4) << cur_frame_id << ".jpg";

  std::string filename = oss.str();//将cur_frame_id转化为字符串放到filename中

  if (!cv::imwrite(filename, cur_img))
  {
    return -1;
  }
  return 0;
}

//初始化随机颜色，存储到randColor[]这个数组中               
static void initializeRandColors()
{
  cv::RNG rng(0xFFFFFFFF); // 生成随机颜色
  for (int i = 0; i < cnum; i++)
  {
    rng.fill(randColor[i], cv::RNG::UNIFORM, 0, 256);
  }
  init_colors = true;
}

// 显示检测结果，找到矩形中心点，画红圆
void show_draw_results(DetectResultsGroup &results_group)
{
  if (!init_colors)
  {
    initializeRandColors();
  }
  char text[256];
  for (const auto& res : results_group.dets)
  {
    sprintf(text, "%s", res.det_name.c_str());
    cv::rectangle(results_group.cur_img, res.box, randColor[1], 2, 8, 0);
    cv::putText(results_group.cur_img, text, cv::Point(res.box.x, res.box.y + 12), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
    
    int cv_center_x = res.box.x + res.box.width / 2;
    int cv_center_y = res.box.y + res.box.height / 2;
    
    cv::circle(results_group.cur_img, cv::Point(cv_center_x, cv_center_y), 10, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
  }
}
