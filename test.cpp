#include <opencv2/opencv.hpp>  
#include <opencv2/superpoint.hpp>  
#include <opencv2/stitching/detail/lightglue.hpp>  
  
int main() {  
    // 测试SuperPoint创建  
    auto superpoint = cv::SuperPoint::create("/root/weights/superpoint.onnx");  
      
    // 测试LightGlue创建    
    auto lightglue = cv::detail::LightGlue::create("/root/weights/superpoint_lightglue.onnx");  
      
    std::cout << "SuperPoint and LightGlue integration successful!" << std::endl;  
    return 0;  
}