#include <stdio.h>
#include <iomanip>
#include <sstream>
#include <mutex>

#include <iostream>

#include "rknn_api.h"  // RKNN API头文件

#include "det/preprocess.h"  
#include "common.h"  

#include "opencv2/core/core.hpp" 
#include "opencv2/highgui/highgui.hpp"  
#include "opencv2/imgproc/imgproc.hpp"  

#include "coreNum.hpp"  // 核心数量相关头文件
#include "rkpt.hpp"  // RKPT类头文件

// 添加到类定义之前
// 定义模型类型
enum ModelType {
    MODEL_MATERIAL = 0, // 物资识别模型
    MODEL_DIGIT = 1     // 数字识别模型
};

///////////////////这个文件主要是把图像转换为他这里的yolo要用的数字形式，并且调用rknn模型进行推理///////////////////////


// 打印张量属性
static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    std::string shape_str = attr->n_dims < 1 ? "" : std::to_string(attr->dims[0]);
    for (int i = 1; i < attr->n_dims; ++i)
    {
        shape_str += ", " + std::to_string(attr->dims[i]);
    }

}

// 从文件中加载数据
static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
    unsigned char *data;
    int ret;

    data = NULL;

    if (NULL == fp)
    {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);  // 设置文件指针位置
    if (ret != 0)
    {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char *)malloc(sz);  // 分配内存
    if (data == NULL)
    {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);  // 读取数据
    return data;
}

// 加载模型文件
static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp;
    unsigned char *data;

    fp = fopen(filename, "rb");  // 打开模型文件
    if (NULL == fp)
    {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);  // 定位到文件末尾
    int size = ftell(fp);  // 获取文件大小

    data = load_data(fp, 0, size);  // 加载文件数据

    fclose(fp);  // 关闭文件

    *model_size = size;  // 设置模型大小
    return data;
}

// RKPT类构造函数
RkPt::RkPt(const std::string &model_path)
{
    this->model_path = model_path;  // 初始化模型路径
    nms_threshold = NMS_THRESH;      // 默认的NMS阈值为0.45
    box_conf_threshold = BOX_THRESH; // 默认的置信度阈值为0.45
    model_type = MODEL_MATERIAL;     // 默认为物资识别模型
}

// RKPT类初始化函数
int RkPt::init(rknn_context *ctx_in, bool share_weight)
{
    printf("Loading model...\n");
    int model_data_size = 0;
    model_data = load_model(model_path.c_str(), &model_data_size);  // 加载模型
    // 模型参数复用
    if (share_weight == true)
        ret = rknn_dup_context(ctx_in, &ctx);  // 复制上下文
    else
        ret = rknn_init(&ctx, model_data, model_data_size, 0, NULL);  // 初始化RKNN上下文
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }

    // 设置模型绑定的核心
    rknn_core_mask core_mask;
    switch (get_core_num())  // 获取核心数量
    {
    case 0:
        core_mask = RKNN_NPU_CORE_0;  // 绑定到核心0
        break;
    case 1:
        core_mask = RKNN_NPU_CORE_1;  // 绑定到核心1
        break;
    case 2:
        core_mask = RKNN_NPU_CORE_2;  // 绑定到核心2
        break;
    }
    ret = rknn_set_core_mask(ctx, core_mask);  // 设置核心掩码
    if (ret < 0)
    {
        printf("rknn_init core error ret=%d\n", ret);
        return -1;
    }

    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));  // 查询SDK版本
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);  // 打印SDK版本信息

    // 获取模型输入输出参数
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));  // 查询输入输出数量
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    // printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // 设置输入参数
    input_attrs = (rknn_tensor_attr *)calloc(io_num.n_input, sizeof(rknn_tensor_attr));  // 分配输入属性内存
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));  // 查询输入属性
        if (ret < 0)
        {
            printf("rknn_init error ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));  // 打印输入属性
    }

    // 设置输出参数
    output_attrs = (rknn_tensor_attr *)calloc(io_num.n_output, sizeof(rknn_tensor_attr));  // 分配输出属性内存
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));  // 查询输出属性
        dump_tensor_attr(&(output_attrs[i]));  // 打印输出属性
    }

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)  // 判断输入格式是否为NCHW
    {
        // printf("model is NCHW input fmt\n");
        channel = input_attrs[0].dims[1];  // 获取通道数
        height = input_attrs[0].dims[2];  // 获取高度
        width = input_attrs[0].dims[3];  // 获取宽度
    }
    else  // 输入格式为NHWC
    {
        // printf("model is NHWC input fmt\n");
        height = input_attrs[0].dims[1];  // 获取高度
        width = input_attrs[0].dims[2];  // 获取宽度
        channel = input_attrs[0].dims[3];  // 获取通道数
    }
    // printf("model input height=%d, width=%d, channel=%d\n", height, width, channel);

    memset(inputs, 0, sizeof(inputs));  // 初始化输入结构体
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;  // 设置输入数据类型
    inputs[0].size = width * height * channel;  // 设置输入数据大小
    inputs[0].fmt = RKNN_TENSOR_NHWC;  // 设置输入数据格式
    inputs[0].pass_through = 0;

    return 0;
}

// 获取RKNN上下文
rknn_context *RkPt::get_pctx()
{
    return &ctx;
}

// 推理函数
DetectResultsGroup RkPt::infer(cv::Mat &orig_img, int cur_frame_id)
{
    std::lock_guard<std::mutex> lock(mtx);  // 加锁，确保线程安全
    cv::Mat img;
    cv::cvtColor(orig_img, img, cv::COLOR_BGR2RGB);  // 将图像转换为RGB格式
    img_width = img.cols;  // 获取图像宽度
    img_height = img.rows;  // 获取图像高度

    BOX_RECT pads;
    memset(&pads, 0, sizeof(BOX_RECT));  // 初始化填充结构体
    cv::Size target_size(width, height);  // 设置目标尺寸
    cv::Mat resized_img(target_size.height, target_size.width, CV_8UC3);  // 创建缩放后的图像
    // 计算缩放比例
    float scale_w = (float)target_size.width / img.cols;
    float scale_h = (float)target_size.height / img.rows;

    // 图像缩放
    if (img_width != width || img_height != height)
    {
        // rga
        rga_buffer_t src;
        rga_buffer_t dst;
        memset(&src, 0, sizeof(src));  // 初始化源缓冲区
        memset(&dst, 0, sizeof(dst));  // 初始化目标缓冲区
        ret = resize_rga(src, dst, img, resized_img, target_size);  // 使用RGA进行图像缩放
        if (ret != 0)
        {
            fprintf(stderr, "resize with rga error\n");
        }

        inputs[0].buf = resized_img.data;  // 设置输入数据缓冲区
    }
    else
    {
        inputs[0].buf = img.data;  // 直接使用原始图像数据
    }

    rknn_inputs_set(ctx, io_num.n_input, inputs);  // 设置输入数据

    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));  // 初始化输出结构体
    for (int i = 0; i < io_num.n_output; i++)
    {
        outputs[i].want_float = 0;  // 设置不需要浮点输出
        outputs[i].is_prealloc = 0;  // 设置不预分配内存
        outputs[i].index = i;  // 设置输出索引
    }

    struct timeval start_time, stop_time;
    gettimeofday(&start_time, NULL);  // 记录开始时间
    ret = rknn_run(ctx, nullptr);  // 运行模型推理
    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);  // 获取输出结果
    gettimeofday(&stop_time, NULL);  // 记录结束时间
    // printf("once run use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);

    // 后处理
    DetectResultsGroup det_result;
    std::vector<int32_t> qnt_zps;  // 量化零点
    std::vector<float> qnt_scales;  // 量化尺度

    for (int i = 0; i < io_num.n_output; i++)
    {
        if (output_attrs[i].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC || output_attrs[i].qnt_type == RKNN_TENSOR_QNT_DFP)
        {
            if (output_attrs[i].type == RKNN_TENSOR_INT8 || output_attrs[i].type == RKNN_TENSOR_UINT8)
            {
                qnt_zps.push_back(output_attrs[i].zp);  // 添加量化零点
                qnt_scales.push_back(output_attrs[i].scale);  // 添加量化尺度
            }
        }
    }

    ret = post_process((int8_t *)outputs[0].buf, (int8_t *)outputs[1].buf, (int8_t *)outputs[2].buf, height, width,
                       box_conf_threshold, nms_threshold, pads, scale_w, scale_h, qnt_zps, qnt_scales, &det_result);  // 后处理
                       
    // 设置结果的当前帧ID和图像
    det_result.cur_frame_id = cur_frame_id;
    det_result.cur_img = orig_img.clone();
    
    // 设置每个检测框的模型类型
    for (auto &det : det_result.dets) {
        det.model_type = this->model_type;
    }

    ret = rknn_outputs_release(ctx, io_num.n_output, outputs);  // 释放输出结果

    return det_result;  // 返回检测结果
}

// RKPT类析构函数
RkPt::~RkPt()
{
    ret = rknn_destroy(ctx);  // 销毁RKNN上下文

    if (model_data)
        free(model_data);  // 释放模型数据

    if (input_attrs)
        free(input_attrs);  // 释放输入属性
    if (output_attrs)
        free(output_attrs);  // 释放输出属性
}
