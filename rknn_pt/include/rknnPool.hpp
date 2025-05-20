#ifndef RKNNPOOL_H
#define RKNNPOOL_H

#include "ThreadPool.hpp"
#include <vector>
#include <iostream>
#include <mutex>
#include <queue>
#include <memory>

//             rknnModel模型类,         模型输入类型              模型输出类型
template <typename rknnModel, typename inputType, typename outputType>
class rknnPool
{
private:
    int threadNum; // 线程数量
    std::string modelPath; // 模型路径

    long long id; // 模型ID
    std::mutex idMtx, queueMtx; // 互斥锁，用于保护id和队列
    std::unique_ptr<dpool::ThreadPool> pool; // 线程池
    std::queue<std::future<outputType>> futs; // 存储推理结果的队列
    std::vector<std::shared_ptr<rknnModel>> models; // 模型实例列表

protected:
    int getModelId(); // 获取模型ID

public:
    rknnPool(const std::string modelPath, int threadNum);// 构造函数，初始化模型路径和线程数量
    int init();                                          // 初始化线程池和模型实例
    int put(inputType inputData, int cur_frame_id);      // 提交输入数据到线程池进行推理
    // int put(inputType inputData);

    int get(outputType &outputData);                     // 从队列中获取推理结果
    ~rknnPool();                                         // 析构函数，释放资源
    
    rknnModel* get_model_ptr();                          // 获取模型指针
};

//构造函数：  传入模型路径、线程数
template <typename rknnModel, typename inputType, typename outputType>
rknnPool<rknnModel, inputType, outputType>::rknnPool(const std::string modelPath, int threadNum)
{
    this->modelPath = modelPath;
    this->threadNum = threadNum;
    this->id = 0;
}

//init函数：  初始化模型、线程池
template <typename rknnModel, typename inputType, typename outputType>
int rknnPool<rknnModel, inputType, outputType>::init()
{
    try
    {
        this->pool = std::make_unique<dpool::ThreadPool>(this->threadNum);   
        for (int i = 0; i < this->threadNum; i++)
            models.push_back(std::make_shared<rknnModel>(this->modelPath.c_str()));
    }
    catch (const std::bad_alloc &e)
    {
        std::cout << "Out of memory: " << e.what() << std::endl;
        return -1;
    }

    // 初始化模型/Initialize the model
    for (int i = 0, ret = 0; i < threadNum; i++)
    {
        ret = models[i]->init(models[0]->get_pctx(), i != 0);
        if (ret != 0)
            return ret;
    }

    return 0;
}

//获取模型id的函数：      利用互斥保护共享资源，并返回模型id
template <typename rknnModel, typename inputType, typename outputType>
int rknnPool<rknnModel, inputType, outputType>::getModelId()
{
    std::lock_guard<std::mutex> lock(idMtx);
    int modelId = id % threadNum;
    id++;
    return modelId;
}

// template <typename rknnModel, typename inputType, typename outputType>
// int rknnPool<rknnModel, inputType, outputType>::put(inputType inputData)
// {
//     std::lock_guard<std::mutex> lock(queueMtx);
//     futs.push(pool->submit(&rknnModel::infer, models[this->getModelId()], inputData));
//     return 0;
// }


//放入的函数，利用互斥保护共享资源，并将模型、输入数据、当前帧号作为参数传入
template <typename rknnModel, typename inputType, typename outputType>
int rknnPool<rknnModel, inputType, outputType>::put(inputType inputData, int cur_frame_id)
// int rknnPool<rknnModel, inputType, outputType>::put(inputType inputData)
{
    std::lock_guard<std::mutex> lock(queueMtx);//利用互斥保护共享资源
    //调用infer函数，                                            并将模型、输入数据、当前帧号作为参数传入
    futs.push(pool->submit(&rknnModel::infer, models[this->getModelId()], inputData, cur_frame_id));
    // futs.push(pool->submit(&rknnModel::infer, models[this->getModelId()], inputData));

    return 0;
}

//取出的函数，利用互斥保护共享资源，并将输出数据作为参数传入
template <typename rknnModel, typename inputType, typename outputType>
int rknnPool<rknnModel, inputType, outputType>::get(outputType &outputData)
{
    std::lock_guard<std::mutex> lock(queueMtx);//利用互斥保护共享资源
    if(futs.empty() == true)
        return 1;
    outputData = futs.front().get();
    futs.pop();
    return 0;
}

//析构函数，清空队列
template <typename rknnModel, typename inputType, typename outputType>
rknnPool<rknnModel, inputType, outputType>::~rknnPool()
{
    while (!futs.empty())
    {
        outputType temp = futs.front().get();
        futs.pop();
    }
}

// 获取模型指针函数
template <typename rknnModel, typename inputType, typename outputType>
rknnModel* rknnPool<rknnModel, inputType, outputType>::get_model_ptr()
{
    if (models.empty()) return nullptr;
    return models[0].get();
}

#endif
