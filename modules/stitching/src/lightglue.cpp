#ifdef HAVE_ONNXRUNTIME
#include "opencv2/stitching/detail/lightglue.hpp"
#include <onnxruntime_cxx_api.h>
#ifdef _WIN32
#include <locale>
#include <codecvt>
#endif
#include <set>
#include <iostream>

namespace cv {
namespace detail {

LightGlue::LightGlue(const String& modelPath, Stitcher::Mode mode, float matchThresh)
    : m_initialized(false), m_matchThresh(matchThresh), m_mode(mode), m_modelPath(modelPath) {}

Ptr<LightGlue> LightGlue::create(const String& modelPath, Stitcher::Mode mode, float matchThresh)
{
    return makePtr<LightGlue>(modelPath, mode, matchThresh);
}

void LightGlue::initializeSession() const
{
    if (m_initialized) return;
    
    m_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_FATAL, "LightGlue");
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

void LightGlue::match(const ImageFeatures& features1, const ImageFeatures& features2,
    MatchesInfo& matches_info)
{
    initializeSession();
    
    // 准备关键点数据
    auto prepareKeypoints = [](const std::vector<KeyPoint>& keypoints, Size img_size) {
        std::vector<float> kp(keypoints.size() * 2);
        float width_center = img_size.width / 2.0f;
        float height_center = img_size.height / 2.0f;
        
        for (size_t i = 0; i < keypoints.size(); ++i) {
            kp[2*i] = (keypoints[i].pt.x - width_center) / width_center;
            kp[2*i + 1] = (keypoints[i].pt.y - height_center) / height_center;
        }
        return kp;
    };
    
    std::vector<float> kp1 = prepareKeypoints(features1.keypoints, features1.img_size);
    std::vector<float> kp2 = prepareKeypoints(features2.keypoints, features2.img_size);
    
    // 准备描述子数据
    auto prepareDescriptors = [](const Mat& descriptors) {
        std::vector<float> desc(descriptors.total());
        Mat descriptorsMat = descriptors.getMat(ACCESS_READ);
        std::memcpy(desc.data(), descriptorsMat.data, descriptors.total() * sizeof(float));
        return desc;
    };
    
    std::vector<float> des1 = prepareDescriptors(features1.descriptors);
    std::vector<float> des2 = prepareDescriptors(features2.descriptors);
    
    // 准备输入张量
    const char* input_names[] = { "kpts0", "kpts1", "desc0", "desc1" };
    std::vector<Ort::Value> inputTensor;
    
    std::vector<int64_t> kp1Shape{ 1, (int64_t)features1.keypoints.size(), 2 };
    inputTensor.emplace_back(Ort::Value::CreateTensor<float>(*m_memoryInfo, kp1.data(), kp1.size(), kp1Shape.data(), kp1Shape.size()));
    
    std::vector<int64_t> kp2Shape{ 1, (int64_t)features2.keypoints.size(), 2 };
    inputTensor.emplace_back(Ort::Value::CreateTensor<float>(*m_memoryInfo, kp2.data(), kp2.size(), kp2Shape.data(), kp2Shape.size()));
    
    std::vector<int64_t> des1Shape{ 1, (int64_t)features1.keypoints.size(), features1.descriptors.cols };
    inputTensor.emplace_back(Ort::Value::CreateTensor<float>(*m_memoryInfo, des1.data(), des1.size(), des1Shape.data(), des1Shape.size()));
    
    std::vector<int64_t> des2Shape{ 1, (int64_t)features2.keypoints.size(), features2.descriptors.cols };
    inputTensor.emplace_back(Ort::Value::CreateTensor<float>(*m_memoryInfo, des2.data(), des2.size(), des2Shape.data(), des2Shape.size()));
    
    // 运行推理
    const char* output_names[] = { "matches0", "matches1", "mscores0", "mscores1" };
    Ort::RunOptions run_options;
    auto outputs = m_session->Run(run_options, input_names, inputTensor.data(), 4, output_names, 4);
    
    // 处理输出
    int64_t* match1 = outputs[0].GetTensorMutableData<int64_t>();
    int64_t* match2 = outputs[1].GetTensorMutableData<int64_t>();
    float* mscore1 = outputs[2].GetTensorMutableData<float>();
    float* mscore2 = outputs[3].GetTensorMutableData<float>();
    
    size_t match1counts = outputs[0].GetTensorTypeAndShapeInfo().GetShape()[1];
    size_t match2counts = outputs[1].GetTensorTypeAndShapeInfo().GetShape()[1];
    
    matches_info.src_img_idx = features1.img_idx;
    matches_info.dst_img_idx = features2.img_idx;
    
    std::set<std::pair<int, int>> matches;
    for (size_t i = 0; i < match1counts; ++i) {
        if (match1[i] > -1 && mscore1[i] > m_matchThresh && match2[match1[i]] == i) {
            matches_info.matches.emplace_back(i, static_cast<int>(match1[i]), 0);
            matches.emplace(i, match1[i]);
        }
    }
    
    for (size_t i = 0; i < match2counts; ++i) {
        if (match2[i] > -1 && mscore2[i] > m_matchThresh && match1[match2[i]] == i) {
            auto match_pair = std::make_pair(static_cast<int>(match2[i]), i);
            if (matches.find(match_pair) == matches.end()) {
                matches_info.matches.emplace_back(match_pair.first, match_pair.second, 0);
            }
        }
    }
    
    std::cout << "matches count:" << matches_info.matches.size() << std::endl;
    
    // 估计变换矩阵
    if (matches_info.matches.size() < 4) {
        matches_info.confidence = 0;
        matches_info.num_inliers = 0;
        return;
    }
    
    Mat src_points(1, static_cast<int>(matches_info.matches.size()), CV_32FC2);
    Mat dst_points(1, static_cast<int>(matches_info.matches.size()), CV_32FC2);
    
    if (m_mode == Stitcher::SCANS) {
        for (size_t i = 0; i < matches_info.matches.size(); ++i) {
            src_points.at<Point2f>(0, static_cast<int>(i)) = features1.keypoints[matches_info.matches[i].queryIdx].pt;
            dst_points.at<Point2f>(0, static_cast<int>(i)) = features2.keypoints[matches_info.matches[i].trainIdx].pt;
        }
        
        matches_info.H = estimateAffine2D(src_points, dst_points, matches_info.inliers_mask);
        if (matches_info.H.empty()) {
            matches_info.confidence = 0;
            matches_info.num_inliers = 0;
            return;
        }
        
        matches_info.num_inliers = cv::countNonZero(matches_info.inliers_mask);
        matches_info.confidence = matches_info.num_inliers / (8 + 0.3 * matches_info.matches.size());
        
        matches_info.H.push_back(Mat::zeros(1, 3, CV_64F));
        matches_info.H.at<double>(2, 2) = 1;
    } else if (m_mode == Stitcher::PANORAMA) {
        for (size_t i = 0; i < matches_info.matches.size(); ++i) {
            Point2f p1 = features1.keypoints[matches_info.matches[i].queryIdx].pt;
            p1 -= Point2f(features1.img_size.width * 0.5f, features1.img_size.height * 0.5f);
            src_points.at<Point2f>(0, static_cast<int>(i)) = p1;
            
            Point2f p2 = features2.keypoints[matches_info.matches[i].trainIdx].pt;
            p2 -= Point2f(features2.img_size.width * 0.5f, features2.img_size.height * 0.5f);
            dst_points.at<Point2f>(0, static_cast<int>(i)) = p2;
        }
        
        matches_info.H = findHomography(src_points, dst_points, matches_info.inliers_mask, RANSAC);
        if (matches_info.H.empty() || std::abs(determinant(matches_info.H)) < std::numeric_limits<double>::epsilon()) {
            matches_info.confidence = 0;
            matches_info.num_inliers = 0;
            return;
        }
        
        matches_info.num_inliers = cv::countNonZero(matches_info.inliers_mask);
        matches_info.confidence = matches_info.num_inliers / (8 + 0.3 * matches_info.matches.size());
        matches_info.confidence = matches_info.confidence > 3. ? 0. : matches_info.confidence;
        
        if (matches_info.num_inliers < 6) return;
        
        // 精炼变换矩阵
        Mat refined_src_points(1, matches_info.num_inliers, CV_32FC2);
        Mat refined_dst_points(1, matches_info.num_inliers, CV_32FC2);
        int inlier_idx = 0;
        for (size_t i = 0; i < matches_info.matches.size(); ++i) {
            if (!matches_info.inliers_mask[i]) continue;
            
            const DMatch& m = matches_info.matches[i];
            refined_src_points.at<Point2f>(0, inlier_idx) = src_points.at<Point2f>(0, i);
            refined_dst_points.at<Point2f>(0, inlier_idx) = dst_points.at<Point2f>(0, i);
            inlier_idx++;
        }
        
        matches_info.H = findHomography(refined_src_points, refined_dst_points, RANSAC);
    }
    
    std::cout << matches_info.H << std::endl;
    AddFeature(features1);
    AddFeature(features2);
    AddMatcheinfo(matches_info);
}

void LightGlue::AddFeature(const ImageFeatures& features)
{
    auto it = std::find_if(features_.begin(), features_.end(),
        [&features](const ImageFeatures& f) { return f.img_idx == features.img_idx; });
    if (it == features_.end()) {
        features_.push_back(features);
    }
}

void LightGlue::AddMatcheinfo(const MatchesInfo& matches)
{
    auto it = std::find_if(pairwise_matches_.begin(), pairwise_matches_.end(),
        [&matches](const MatchesInfo& m) {
            return (m.src_img_idx == matches.src_img_idx && m.dst_img_idx == matches.dst_img_idx) ||
                   (m.src_img_idx == matches.dst_img_idx && m.dst_img_idx == matches.src_img_idx);
        });
    if (it == pairwise_matches_.end()) {
        pairwise_matches_.push_back(matches);
    }
}

} // namespace detail
} // namespace cv
#endif // HAVE_ONNXRUNTIME