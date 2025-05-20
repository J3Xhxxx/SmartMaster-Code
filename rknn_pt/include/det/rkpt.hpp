#ifndef RKPT_H
#define RKPT_H

#include "rknn_api.h"

#include "opencv2/core/core.hpp"
#include "postprocess.h"

// 定义模型类型
enum ModelType {
    MODEL_MATERIAL = 0, // 物资识别模型
    MODEL_DIGIT = 1     // 数字识别模型
};

static void dump_tensor_attr(rknn_tensor_attr *attr);
static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz);
static unsigned char *load_model(const char *filename, int *model_size);
static int saveFloat(const char *file_name, float *output, int element_size);

class RkPt
{
private:
    int ret;
    std::mutex mtx;
    std::string model_path;
    unsigned char *model_data;

    rknn_context ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr *input_attrs;
    rknn_tensor_attr *output_attrs;
    rknn_input inputs[1];

    int channel, width, height;
    int img_width, img_height;

    float nms_threshold, box_conf_threshold;
    int model_type; // 模型类型：0为物资模型，1为数字模型

public:
    RkPt(const std::string &model_path);
    int init(rknn_context *ctx_in, bool isChild);
    rknn_context *get_pctx();
    // cv::Mat infer(cv::Mat &ori_img);
    // std::vector<detect_result_t> infer(cv::Mat &ori_img);
    DetectResultsGroup infer(cv::Mat &ori_img, int cur_frame_id);
    
    // 设置置信度和NMS阈值
    void set_thresholds(float conf_thresh, float nms_thresh) {
        box_conf_threshold = conf_thresh;
        nms_threshold = nms_thresh;
    }
    
    // 获取当前设置的阈值
    float get_conf_threshold() const { return box_conf_threshold; }
    float get_nms_threshold() const { return nms_threshold; }
    
    // 设置和获取模型类型
    void set_model_type(int type) { model_type = type; }
    int get_model_type() const { return model_type; }
    
    ~RkPt();
};

#endif
