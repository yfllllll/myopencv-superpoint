#ifndef OPENCV_STITCHING_DETAIL_LIGHTGLUE_HPP
#define OPENCV_STITCHING_DETAIL_LIGHTGLUE_HPP

#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching.hpp"
#include "opencv2/core.hpp"

#ifdef HAVE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

#include <iostream>
#include <memory>
#include <vector>

namespace cv {
namespace detail {

/**
 * @brief LightGlue feature matcher.
 *
 * LightGlue is a lightweight feature matcher with high accuracy, based on adaptive pruning
 * of keypoint and match candidates. It works particularly well with SuperPoint features.
 *
 * @note This implementation requires ONNX Runtime and a pre-trained LightGlue model file.
 */
class CV_EXPORTS_W LightGlue : public FeaturesMatcher
{
public:
    /**
     * @brief Constructor
     * @param modelPath Path to the LightGlue ONNX model file
     * @param mode Stitching mode (PANORAMA for perspective, SCANS for affine transformation)
     * @param matchThresh Confidence threshold for match validation (0.0 means no threshold)
     */
    CV_WRAP LightGlue(const String& modelPath = String(),
                     Stitcher::Mode mode = Stitcher::PANORAMA,
                     float matchThresh = 0.0f);

    /** @brief Destructor */
    virtual ~LightGlue() = default;

    /**
     * @brief Creates a LightGlue feature matcher
     * @param modelPath Path to the LightGlue ONNX model file
     * @param mode Stitching mode (PANORAMA for perspective, SCANS for affine transformation)
     * @param matchThresh Confidence threshold for match validation
     * @return Pointer to the created LightGlue instance
     */
    CV_WRAP static Ptr<LightGlue> create(const String& modelPath = String(),
                                        Stitcher::Mode mode = Stitcher::PANORAMA,
                                        float matchThresh = 0.0f);

    /**
     * @brief Matches features between two images
     * @param features1 Features from the first image
     * @param features2 Features from the second image
     * @param matches_info Output matching information including homography
     */
    CV_WRAP void match(const ImageFeatures& features1, const ImageFeatures& features2,
                      MatchesInfo& matches_info) CV_OVERRIDE;

    /**
     * @brief Functional interface for matching (calls match internally)
     * @param features1 Features from the first image
     * @param features2 Features from the second image
     * @param matches_info Output matching information
     */
    CV_WRAP_AS(apply) void operator()(const ImageFeatures& features1, const ImageFeatures& features2,
                                     CV_OUT MatchesInfo& matches_info)
    {
        match(features1, features2, matches_info);
    }

    /**
     * @brief Get cached features from all processed images
     * @return Vector of all processed image features
     */
    CV_WRAP std::vector<ImageFeatures> features() const { return features_; }

    /**
     * @brief Get cached match information from all processed image pairs
     * @return Vector of all computed pairwise matches
     */
    CV_WRAP std::vector<MatchesInfo> matchinfo() const { return pairwise_matches_; }

protected:
    Stitcher::Mode m_mode;              ///< Transformation mode (affine or perspective)
    String m_modelPath;                 ///< Path to ONNX model file
    std::vector<ImageFeatures> features_;           ///< Cache of processed image features
    std::vector<MatchesInfo> pairwise_matches_;     ///< Cache of computed matches
    float m_matchThresh;                ///< Match confidence threshold

    // ONNX Runtime session management members
#ifdef HAVE_ONNXRUNTIME
    mutable std::unique_ptr<Ort::Env> m_env;
    mutable std::unique_ptr<Ort::Session> m_session;
    mutable std::unique_ptr<Ort::MemoryInfo> m_memoryInfo;
    mutable bool m_initialized;
#endif

    /** @brief Initialize ONNX Runtime session */
    void initializeSession() const;

    /**
     * @brief Adds image features to internal cache (prevents duplicates)
     * @param features Image features to add
     */
    void AddFeature(const ImageFeatures& features);

    /**
     * @brief Adds match information to internal cache (prevents duplicates)
     * @param matches_info Match information to add
     */
    void AddMatcheinfo(const MatchesInfo& matches_info);

#ifdef HAVE_ONNXRUNTIME
    /**
     * @brief Create ONNX Runtime session options
     * @return Configured session options
     */
    Ort::SessionOptions createSessionOptions();

    /**
     * @brief Create memory information object
     * @return Configured memory information
     */
    Ort::MemoryInfo createMemoryInfo();
#endif
};

} // namespace detail
} // namespace cv

#endif // OPENCV_STITCHING_DETAIL_LIGHTGLUE_HPP