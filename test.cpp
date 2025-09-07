#include <opencv2/opencv.hpp>  
#include <opencv2/superpoint.hpp>  
#include <opencv2/stitching/detail/lightglue.hpp>  
#include <filesystem>  
#include <iostream>  

int main(int argc, char* argv[]) {  
    if (argc != 2) {  
        std::cout << "Usage: " << argv[0] << " <image_folder_path>" << std::endl;  
        return -1;  
    }  
      
    std::string folder_path = argv[1];  
    std::vector<cv::Mat> images;  
      
    // 读取文件夹中的所有图像  
    std::vector<std::string> image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"};  
      
    for (const auto& entry : std::filesystem::directory_iterator(folder_path)) {  
        if (entry.is_regular_file()) {  
            std::string ext = entry.path().extension().string();  
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);  
              
            if (std::find(image_extensions.begin(), image_extensions.end(), ext) != image_extensions.end()) {  
                cv::Mat img = cv::imread(entry.path().string());  
                if (!img.empty()) {  
                    images.push_back(img);  
                    std::cout << "Loaded: " << entry.path().filename() << std::endl;  
                } else {  
                    std::cout << "Failed to load: " << entry.path().filename() << std::endl;  
                }  
            }  
        }  
    }  
      
    if (images.size() < 2) {  
        std::cout << "Need at least 2 images for stitching. Found: " << images.size() << std::endl;  
        return -1;  
    }  
      
    std::cout << "Found " << images.size() << " images for stitching." << std::endl;  
      
    // 创建Stitcher并配置SuperPoint和LightGlue  
    auto stitcher = cv::Stitcher::create(cv::Stitcher::PANORAMA);  
      
    // 设置SuperPoint特征检测器  
    auto superpoint = cv::SuperPoint::create("/root/weights/superpoint.onnx");  
    stitcher->setFeaturesFinder(superpoint);  
      
    // 设置LightGlue匹配器  
    auto lightglue = cv::detail::LightGlue::create("/root/weights/superpoint_lightglue.onnx");  
    stitcher->setFeaturesMatcher(lightglue);  
      
    std::cout << "SuperPoint and LightGlue configured successfully!" << std::endl;  
      
    // 执行拼接  
    cv::Mat pano;  
    cv::Stitcher::Status status = stitcher->stitch(images, pano);  
      
    if (status != cv::Stitcher::OK) {  
        std::cout << "Can't stitch images, error code = " << int(status) << std::endl;  
        return -1;  
    }  
      
    // 保存结果  
    std::string output_path = "panorama_result.jpg";  
    cv::imwrite(output_path, pano);  
    std::cout << "Stitching completed successfully! Saved to: " << output_path << std::endl;  
      
    return 0;  
}