#ifndef COMMON_H
#define COMMON_H

#include <opencv2/core.hpp>

#define OBJ_NUMB_MAX_SIZE 64

typedef struct _BOX_RECT
{
    int left;
    int right;
    int top;
    int bottom;
} BOX_RECT;

//这个是绘制矩形的函数用到的结构体
typedef struct _DetectionBox
{
    float score;
    std::string det_name;
    int obj_id;  // 物体分类ID
    int model_type; // 0: 物资识别模型, 1: 数字识别模型
    cv::Rect_<int> box;
}DetectionBox;



// typedef struct _DetectionBox
// {
//     float score;
//     std::string det_name;
//     BOX_RECT box;
// }DetectionBox;


//这个是绘制红心的函数用到的结构体
typedef struct _DetectResultsGroup
{
    cv::Mat cur_img;
    int cur_frame_id;
    std::vector<DetectionBox> dets; // 修改为vector
} DetectResultsGroup;

// typedef struct _DetectResultsGroup
// {
//     cv::Mat cur_img;
//     int cur_frame_id;
//     int count;
//     DetectionBox results[OBJ_NUMB_MAX_SIZE];
// } DetectResultsGroup;


#endif // COMMON_H
