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
    : m_mode(mode), m_modelPath(modelPath), m_matchThresh(matchThresh), m_initialized(false)
{
}

Ptr<LightGlue> LightGlue::create(const String& modelPath, Stitcher::Mode mode, float matchThresh)
{
    return makePtr<LightGlue>(modelPath, mode, matchThresh);
}

Ort::SessionOptions LightGlue::createSessionOptions() const {
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(4);
    options.SetInterOpNumThreads(4);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    try {
        OrtCUDAProviderOptions cuda_options{};
        cuda_options.device_id = 0;
        options.AppendExecutionProvider_CUDA(cuda_options);
        std::cout << "LightGlue: Using GPU (CUDA) acceleration" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "LightGlue: CUDA not available, falling back to CPU: " << e.what() << std::endl;
    }
    return options;
}

Ort::MemoryInfo LightGlue::createMemoryInfo() const {
    return Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
}

void LightGlue::initialize() const {
    if (m_initialized) {
        return;
    }
    m_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "LightGlue");
    Ort::SessionOptions sessionOptions = createSessionOptions();

#ifdef _WIN32
    std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
    std::wstring modelPathW = converter.from_bytes(m_modelPath.c_str());
    m_session = std::make_unique<Ort::Session>(*m_env, modelPathW.c_str(), sessionOptions);
#else
    m_session = std::make_unique<Ort::Session>(*m_env, m_modelPath.c_str(), sessionOptions);
#endif
    m_initialized = true;
}

void LightGlue::match(const ImageFeatures& features1, const ImageFeatures& features2,
    MatchesInfo& matches_info)
{
    initialize();

    std::vector<float> kp1(features1.keypoints.size() * 2);
    float f1wid = features1.img_size.width / 2.0f;
    float f1hei = features1.img_size.height / 2.0f;
    for (size_t i = 0; i < features1.keypoints.size(); i++)
    {
        kp1[2 * i] = (features1.keypoints[i].pt.x - f1wid) / f1wid;
        kp1[2 * i + 1] = (features1.keypoints[i].pt.y - f1hei) / f1hei;
    }

    std::vector<float> kp2(features2.keypoints.size() * 2);
    float f2wid = features2.img_size.width / 2.0f;
    float f2hei = features2.img_size.height / 2.0f;
    for (size_t i = 0; i < features2.keypoints.size(); i++)
    {
        kp2[2 * i] = (features2.keypoints[i].pt.x - f2wid) / f2wid;
        kp2[2 * i + 1] = (features2.keypoints[i].pt.y - f2hei) / f2hei;
    }
    
    // Optimized descriptor flattening
    Mat des1mat = features1.descriptors.getMat(ACCESS_READ);
    std::vector<float> des1(des1mat.rows * des1mat.cols);
    if (des1mat.isContinuous()) {
        memcpy(des1.data(), des1mat.data, des1.size() * sizeof(float));
    } else {
        for (int i = 0; i < des1mat.rows; ++i) {
            memcpy(des1.data() + i * des1mat.cols, des1mat.ptr<float>(i), des1mat.cols * sizeof(float));
        }
    }
    
    Mat des2mat = features2.descriptors.getMat(ACCESS_READ);
    std::vector<float> des2(des2mat.rows * des2mat.cols);
    if (des2mat.isContinuous()) {
        memcpy(des2.data(), des2mat.data, des2.size() * sizeof(float));
    } else {
        for (int i = 0; i < des2mat.rows; ++i) {
            memcpy(des2.data() + i * des2mat.cols, des2mat.ptr<float>(i), des2mat.cols * sizeof(float));
        }
    }

    const char* input_names[] = { "kpts0", "kpts1", "desc0", "desc1" };
    Ort::MemoryInfo memoryInfo = createMemoryInfo();

    std::vector<Ort::Value> inputTensor;
    std::vector<int64_t> kp1Shape{ 1, (int64_t)features1.keypoints.size(), 2 };
    inputTensor.emplace_back(Ort::Value::CreateTensor<float>(memoryInfo, kp1.data(), kp1.size(), kp1Shape.data(), kp1Shape.size()));
    std::vector<int64_t> kp2Shape{ 1, (int64_t)features2.keypoints.size(), 2 };
    inputTensor.emplace_back(Ort::Value::CreateTensor<float>(memoryInfo, kp2.data(), kp2.size(), kp2Shape.data(), kp2Shape.size()));
    std::vector<int64_t> des1Shape{ 1, (int64_t)features1.keypoints.size(), (int64_t)des1mat.cols };
    inputTensor.emplace_back(Ort::Value::CreateTensor<float>(memoryInfo, des1.data(), des1.size(), des1Shape.data(), des1Shape.size()));
    std::vector<int64_t> des2Shape{ 1, (int64_t)features2.keypoints.size(), (int64_t)des2mat.cols };
    inputTensor.emplace_back(Ort::Value::CreateTensor<float>(memoryInfo, des2.data(), des2.size(), des2Shape.data(), des2Shape.size()));

    const char* output_names[] = { "matches0","matches1","mscores0","mscores1" };
    Ort::RunOptions run_options;
    std::vector<Ort::Value> outputs = m_session->Run(run_options, input_names, inputTensor.data(), 4, output_names, 4);

    // ... The rest of the matching logic remains the same ...
    
    int64_t* match1 = (int64_t*)outputs[0].GetTensorMutableData<void>();
    int match1counts = static_cast<int>(outputs[0].GetTensorTypeAndShapeInfo().GetShape()[1]);
    
    float* mscore1 = (float*)outputs[2].GetTensorMutableData<void>();
    
    int64_t* match2 = (int64_t*)outputs[1].GetTensorMutableData<void>();
    int match2counts = static_cast<int>(outputs[1].GetTensorTypeAndShapeInfo().GetShape()[1]);

    float* mscore2 = (float*)outputs[3].GetTensorMutableData<void>();

    matches_info.src_img_idx = features1.img_idx;
    matches_info.dst_img_idx = features2.img_idx;
    matches_info.matches.clear();

    std::set<std::pair<int, int> > matches;
    for (int i = 0; i < match1counts; i++)
    {
        if (match1[i] > -1 && mscore1[i] > this->m_matchThresh && match2[match1[i]] == i)
        {
            DMatch mt;
            mt.queryIdx = i;
            mt.trainIdx = static_cast<int>(match1[i]);
            matches_info.matches.push_back(mt);
            matches.insert(std::make_pair(mt.queryIdx, mt.trainIdx));
        }
    }

    for (int i = 0; i < match2counts; i++)
    {
        if (match2[i] > -1 && mscore2[i] > this->m_matchThresh && match1[match2[i]] == i)
        {
            DMatch mt;
            mt.queryIdx = static_cast<int>(match2[i]);
            mt.trainIdx = i;

            if (matches.find(std::make_pair(mt.queryIdx, mt.trainIdx)) == matches.end())
                matches_info.matches.push_back(mt);
        }
    }

    // ... The rest of the homography estimation logic remains the same ...
    if (matches_info.matches.size() < 4) { // Not enough matches for homography/affine
        matches_info.confidence = 0;
        matches_info.num_inliers = 0;
        matches_info.H.release();
        return;
    }

    Mat src_points(1, static_cast<int>(matches_info.matches.size()), CV_32FC2);
    Mat dst_points(1, static_cast<int>(matches_info.matches.size()), CV_32FC2);
    
    if (this->m_mode == Stitcher::SCANS)
    {
        for (size_t i = 0; i < matches_info.matches.size(); ++i)
        {
            src_points.at<Point2f>(0, static_cast<int>(i)) = features1.keypoints[matches_info.matches[i].queryIdx].pt;
            dst_points.at<Point2f>(0, static_cast<int>(i)) = features2.keypoints[matches_info.matches[i].trainIdx].pt;
        }

        matches_info.H = estimateAffine2D(src_points, dst_points, matches_info.inliers_mask);

        if (matches_info.H.empty()) {
            matches_info.confidence = 0;
            matches_info.num_inliers = 0;
            return;
        }

        matches_info.num_inliers = 0;
        for (size_t i = 0; i < matches_info.inliers_mask.size(); ++i)
            if (matches_info.inliers_mask[i])
                matches_info.num_inliers++;
        
        matches_info.confidence = matches_info.num_inliers / (8 + 0.3 * matches_info.matches.size());
        
        matches_info.H.push_back(Mat::zeros(1, 3, CV_64F));
        matches_info.H.at<double>(2, 2) = 1;
    }
    else if (this->m_mode == Stitcher::PANORAMA)
    {
        for (size_t i = 0; i < matches_info.matches.size(); ++i)
        {
            const DMatch& m = matches_info.matches[i];

            Point2f p = features1.keypoints[m.queryIdx].pt;
            p.x -= features1.img_size.width * 0.5f;
            p.y -= features1.img_size.height * 0.5f;
            src_points.at<Point2f>(0, static_cast<int>(i)) = p;

            p = features2.keypoints[m.trainIdx].pt;
            p.x -= features2.img_size.width * 0.5f;
            p.y -= features2.img_size.height * 0.5f;
            dst_points.at<Point2f>(0, static_cast<int>(i)) = p;
        }

        matches_info.H = findHomography(src_points, dst_points, matches_info.inliers_mask, RANSAC);
        if (matches_info.H.empty() || std::abs(determinant(matches_info.H)) < std::numeric_limits<double>::epsilon())
            return;

        matches_info.num_inliers = 0;
        for (size_t i = 0; i < matches_info.inliers_mask.size(); ++i)
            if (matches_info.inliers_mask[i])
                matches_info.num_inliers++;

        matches_info.confidence = matches_info.num_inliers / (8 + 0.3 * matches_info.matches.size());
        matches_info.confidence = matches_info.confidence > 3. ? 0. : matches_info.confidence;

        if (matches_info.num_inliers < 6)
            return;

        src_points.create(1, matches_info.num_inliers, CV_32FC2);
        dst_points.create(1, matches_info.num_inliers, CV_32FC2);
        int inlier_idx = 0;
        for (size_t i = 0; i < matches_info.matches.size(); ++i)
        {
            if (!matches_info.inliers_mask[i])
                continue;

            const DMatch& m = matches_info.matches[i];

            Point2f p = features1.keypoints[m.queryIdx].pt;
            p.x -= features1.img_size.width * 0.5f;
            p.y -= features1.img_size.height * 0.5f;
            src_points.at<Point2f>(0, inlier_idx) = p;

            p = features2.keypoints[m.trainIdx].pt;
            p.x -= features2.img_size.width * 0.5f;
            p.y -= features2.img_size.height * 0.5f;
            dst_points.at<Point2f>(0, inlier_idx) = p;

            inlier_idx++;
        }
        matches_info.H = findHomography(src_points, dst_points, RANSAC);
    }

    this->AddFeature(features1);
    this->AddFeature(features2);
    this->AddMatcheinfo(matches_info);
}

void LightGlue::AddFeature(const ImageFeatures& features)
{
    for (size_t i = 0; i < this->features_.size(); i++)
    {
        if (features.img_idx == this->features_[i].img_idx)
            return;
    }
    this->features_.push_back(features);
}

void LightGlue::AddMatcheinfo(const MatchesInfo& matches)
{
    for (size_t i = 0; i < this->pairwise_matches_.size(); i++)
    {
        if (matches.src_img_idx == this->pairwise_matches_[i].src_img_idx &&
            matches.dst_img_idx == this->pairwise_matches_[i].dst_img_idx)
            return;
        if (matches.src_img_idx == this->pairwise_matches_[i].dst_img_idx &&
            matches.dst_img_idx == this->pairwise_matches_[i].src_img_idx)
            return;
    }
    this->pairwise_matches_.push_back(MatchesInfo(matches));
}

} // namespace detail
} // namespace cv
#endif // HAVE_ONNXRUNTIME