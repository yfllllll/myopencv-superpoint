#ifndef OPENCV_STITCHING_DETAIL_LIGHTGLUE_HPP
#define OPENCV_STITCHING_DETAIL_LIGHTGLUE_HPP

#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include <iostream>
#include <memory>

#ifdef HAVE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace cv {
namespace detail {

/** @brief LightGlue feature matcher.
*/
class CV_EXPORTS_W LightGlue : public FeaturesMatcher
{
public:
    CV_WRAP LightGlue(const String& modelPath = String(),
                      Stitcher::Mode mode = Stitcher::PANORAMA,
                      float matchThresh = 0.0f);

    CV_WRAP static Ptr<LightGlue> create(const String& modelPath = String(),
                                         Stitcher::Mode mode = Stitcher::PANORAMA,
                                         float matchThresh = 0.0f);

    CV_WRAP void match(const ImageFeatures& features1, const ImageFeatures& features2,
                       MatchesInfo& matches_info) CV_OVERRIDE;

    CV_WRAP_AS(apply) void operator()(const ImageFeatures& features1, const ImageFeatures& features2,
        CV_OUT MatchesInfo& matches_info) {
        match(features1, features2, matches_info);
    }

    CV_WRAP std::vector<ImageFeatures> features() const { return features_; }
    CV_WRAP std::vector<MatchesInfo> matchinfo() const { return pairwise_matches_; }

protected:
    Stitcher::Mode m_mode;
    String m_modelPath;
    std::vector<ImageFeatures> features_;
    std::vector<MatchesInfo> pairwise_matches_;
    float m_matchThresh;

    void AddFeature(const ImageFeatures& features);
    void AddMatcheinfo(const MatchesInfo& matches_info);
    
#ifdef HAVE_ONNXRUNTIME
    // ONNX Runtime session management members
    mutable std::unique_ptr<Ort::Env> m_env;
    mutable std::unique_ptr<Ort::Session> m_session;
    mutable bool m_initialized;

    /** @brief Initializes the ONNX Runtime session if not already done. */
    void initialize() const;
    
    /** @brief Creates and configures ONNX Runtime session options. */
    Ort::SessionOptions createSessionOptions() const;

    /** @brief Creates an ONNX Runtime memory info object. */
    Ort::MemoryInfo createMemoryInfo() const;
#endif
};

} // namespace detail
} // namespace cv

#endif // OPENCV_STITCHING_DETAIL_LIGHTGLUE_HPP