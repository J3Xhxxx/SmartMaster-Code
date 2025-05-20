#ifndef POSTPROCESS_H_
#define POSTPROCESS_H_

#include <stdint.h>
#include <vector>

#include <iomanip>
#include <sstream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "common.h"

#define OBJ_NAME_MAX_SIZE 16
#define OBJ_NUMB_MAX_SIZE 64
#define OBJ_CLASS_NUM 25          // 总类别数：15个物资类别 + 10个数字类别
#define NMS_THRESH 0.45           // 非极大值抑制阈值，可以调整
#define BOX_THRESH 0.65           // 物体检测框置信度阈值，可以调整
#define PROP_BOX_SIZE (5 + OBJ_CLASS_NUM)

// 物体和数字识别的类别数量定义
#define OBJ_MATERIAL_CLASS_NUM 15 // 物资类别数量
#define OBJ_DIGIT_CLASS_NUM 10    // 数字类别数量

int post_process(int8_t *input0, int8_t *input1, int8_t *input2, int model_in_h, int model_in_w,
                 float conf_threshold, float nms_threshold, BOX_RECT pads, float scale_w, float scale_h,
                 std::vector<int32_t> &qnt_zps, std::vector<float> &qnt_scales,
                 DetectResultsGroup *group);

void deinitPostProcess();

int draw_image_detect(cv::Mat &cur_img, std::vector<DetectionBox> &results, int cur_frame_id);

void show_draw_results(DetectResultsGroup &results_group);


#endif //POSTPROCESS_H_
