#ifdef HAVE_ONNXRUNTIME
#include "opencv2/superpoint.hpp"
#include <onnxruntime_cxx_api.h>
#ifdef _WIN32
#include <locale>
#include <codecvt>
#endif

namespace cv {

SuperPoint::SuperPoint(const String& modelPath) : m_initialized(false), m_modelPath(modelPath) {}

Ptr<SuperPoint> SuperPoint::create(const String& modelPath)
{
    return makePtr<SuperPoint>(modelPath);
}

void SuperPoint::initializeSession() const
{
    if (m_initialized) return;
    
    m_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "SuperPoint");
    Ort::SessionOptions sessionOptions = createSessionOptions();
    
#ifdef _WIN32
    std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
    std::wstring modelPathW = converter.from_bytes(m_modelPath.c_str());
    m_session = std::make_unique<Ort::Session>(*m_env, modelPathW.c_str(), sessionOptions);
#else
    m_session = std::make_unique<Ort::Session>(*m_env, m_modelPath.c_str(), sessionOptions);
#endif

    m_memoryInfo = std::make_unique<Ort::MemoryInfo>(createMemoryInfo());
    m_initialized = true;
}

std::vector<float> SuperPoint::ApplyTransform(const Mat& image, float& mean, float& std)
{
    Mat floatImage;
    image.convertTo(floatImage, CV_32FC1);
    
    std::vector<float> imgData;
    imgData.reserve(image.total());
    
    for (int h = 0; h < image.rows; h++)
    {
        for (int w = 0; w < image.cols; w++)
        {
            imgData.push_back(floatImage.at<float>(h, w) / 255.0f);
        }
    }
    return imgData;
}

void SuperPoint::detectAndCompute(InputArray image, InputArray mask,
    std::vector<KeyPoint>& keypoints,
    OutputArray descriptors,
    bool useProvidedKeypoints)
{
    if (useProvidedKeypoints) {
        // 如果使用提供的关键点，只计算描述符
        compute(image, keypoints, descriptors);
        return;
    }
    
    // 确保会话已初始化
    initializeSession();
    
    Mat img = image.getMat();
    Mat grayImg;
    if (img.channels() == 3)
        cvtColor(img, grayImg, COLOR_BGR2GRAY);
    else
        grayImg = img;
        
    // 预处理图像
    float mean, std;
    std::vector<float> imgData = ApplyTransform(grayImg, mean, std);
    std::vector<int64_t> inputShape{1, 1, grayImg.rows, grayImg.cols};
    
    // 创建输入张量
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(*m_memoryInfo, imgData.data(), imgData.size(), 
                                                           inputShape.data(), inputShape.size());
    
    const char* input_names[] = {"image"};
    const char* output_names[] = {"keypoints", "scores", "descriptors"};
    Ort::RunOptions run_options;
    
    // 运行推理
    std::vector<Ort::Value> outputs = m_session->Run(run_options, input_names, &inputTensor, 1, output_names, 3);
    
    // 处理关键点输出
    auto kp_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t* kp_data = outputs[0].GetTensorMutableData<int64_t>();
    int keypoint_count = static_cast<int>(kp_shape[1]);
    
    keypoints.resize(keypoint_count);
    for (int i = 0; i < keypoint_count; i++)
    {
        KeyPoint p;
        p.pt.x = static_cast<float>(kp_data[i * 2]);
        p.pt.y = static_cast<float>(kp_data[i * 2 + 1]);
        keypoints[i] = p;
    }
    
    // 处理描述符输出
    auto desc_shape = outputs[2].GetTensorTypeAndShapeInfo().GetShape();
    float* desc_data = outputs[2].GetTensorMutableData<float>();
    
    Mat desc_mat(static_cast<int>(desc_shape[1]), static_cast<int>(desc_shape[2]), CV_32FC1, desc_data);
    
    // 应用掩码（如果提供）
    if (!mask.empty()) {
        Mat mask_mat = mask.getMat();
        
        // 过滤关键点
        std::vector<KeyPoint> filtered_keypoints;
        std::vector<int> valid_indices;
        
        for (int i = 0; i < keypoints.size(); i++) {
            const KeyPoint& kp = keypoints[i];
            Point pt(static_cast<int>(kp.pt.x + 0.5f), static_cast<int>(kp.pt.y + 0.5f));
            
            // 检查点是否在图像范围内且掩码值不为零
            if (pt.x >= 0 && pt.x < mask_mat.cols && 
                pt.y >= 0 && pt.y < mask_mat.rows &&
                mask_mat.at<uchar>(pt) != 0) {
                filtered_keypoints.push_back(kp);
                valid_indices.push_back(i);
            }
        }
        
        // 更新关键点
        keypoints = std::move(filtered_keypoints);
        
        // 过滤描述符
        if (descriptors.needed() && !valid_indices.empty()) {
            Mat filtered_descriptors(static_cast<int>(valid_indices.size()), desc_mat.cols, desc_mat.type());
            for (int i = 0; i < valid_indices.size(); i++) {
                desc_mat.row(valid_indices[i]).copyTo(filtered_descriptors.row(i));
            }
            filtered_descriptors.copyTo(descriptors);
        } else if (descriptors.needed()) {
            descriptors.release();
        }
    } else {
        // 没有掩码，直接复制描述符
        desc_mat.copyTo(descriptors);
    }
}

void SuperPoint::detect(InputArray image,
    std::vector<KeyPoint>& keypoints,
    InputArray mask)
{
    // 确保会话已初始化
    initializeSession();
    
    Mat img = image.getMat();
    Mat grayImg;
    if (img.channels() == 3)
        cvtColor(img, grayImg, COLOR_BGR2GRAY);
    else
        grayImg = img;
        
    // 预处理图像
    float mean, std;
    std::vector<float> imgData = ApplyTransform(grayImg, mean, std);
    std::vector<int64_t> inputShape{1, 1, grayImg.rows, grayImg.cols};
    
    // 创建输入张量
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(*m_memoryInfo, imgData.data(), imgData.size(), 
                                                           inputShape.data(), inputShape.size());
    
    const char* input_names[] = {"image"};
    const char* output_names[] = {"keypoints", "scores", "descriptors"};
    Ort::RunOptions run_options;
    
    // 运行推理
    std::vector<Ort::Value> outputs = m_session->Run(run_options, input_names, &inputTensor, 1, output_names, 3);
    
    // 处理关键点输出
    auto kp_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t* kp_data = outputs[0].GetTensorMutableData<int64_t>();
    int keypoint_count = static_cast<int>(kp_shape[1]);
    
    keypoints.resize(keypoint_count);
    for (int i = 0; i < keypoint_count; i++)
    {
        KeyPoint p;
        p.pt.x = static_cast<float>(kp_data[i * 2]);
        p.pt.y = static_cast<float>(kp_data[i * 2 + 1]);
        keypoints[i] = p;
    }
    
    // 应用掩码（如果提供）
    if (!mask.empty()) {
        Mat mask_mat = mask.getMat();
        std::vector<KeyPoint> filtered_keypoints;
        
        for (const auto& kp : keypoints) {
            Point pt(static_cast<int>(kp.pt.x + 0.5f), static_cast<int>(kp.pt.y + 0.5f));
            
            // 检查点是否在图像范围内且掩码值不为零
            if (pt.x >= 0 && pt.x < mask_mat.cols && 
                pt.y >= 0 && pt.y < mask_mat.rows &&
                mask_mat.at<uchar>(pt) != 0) {
                filtered_keypoints.push_back(kp);
            }
        }
        
        // 更新关键点
        keypoints = std::move(filtered_keypoints);
    }
}

void SuperPoint::compute(InputArray image,
    std::vector<KeyPoint>& keypoints,
    OutputArray descriptors)
{
    // 确保会话已初始化
    initializeSession();
    
    Mat img = image.getMat();
    Mat grayImg;
    if (img.channels() == 3)
        cvtColor(img, grayImg, COLOR_BGR2GRAY);
    else
        grayImg = img;
        
    // 预处理图像
    float mean, std;
    std::vector<float> imgData = ApplyTransform(grayImg, mean, std);
    std::vector<int64_t> inputShape{1, 1, grayImg.rows, grayImg.cols};
    
    // 创建输入张量
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(*m_memoryInfo, imgData.data(), imgData.size(), 
                                                           inputShape.data(), inputShape.size());
    
    const char* input_names[] = {"image"};
    const char* output_names[] = {"keypoints", "scores", "descriptors"};
    Ort::RunOptions run_options;
    
    // 运行推理
    std::vector<Ort::Value> outputs = m_session->Run(run_options, input_names, &inputTensor, 1, output_names, 3);
    
    // 处理描述符输出
    auto desc_shape = outputs[2].GetTensorTypeAndShapeInfo().GetShape();
    float* desc_data = outputs[2].GetTensorMutableData<float>();
    
    Mat desc_mat(static_cast<int>(desc_shape[1]), static_cast<int>(desc_shape[2]), CV_32FC1, desc_data);
    
    // 如果关键点数量与描述符数量不匹配，更新关键点
    if (keypoints.size() != desc_mat.rows) {
        // 处理关键点输出
        auto kp_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        int64_t* kp_data = outputs[0].GetTensorMutableData<int64_t>();
        int keypoint_count = static_cast<int>(kp_shape[1]);
        
        keypoints.resize(keypoint_count);
        for (int i = 0; i < keypoint_count; i++)
        {
            KeyPoint p;
            p.pt.x = static_cast<float>(kp_data[i * 2]);
            p.pt.y = static_cast<float>(kp_data[i * 2 + 1]);
            keypoints[i] = p;
        }
    }
    
    // 复制描述符到输出
    desc_mat.copyTo(descriptors);
}

} // namespace cv
#endif // HAVE_ONNXRUNTIME