import cv2
import numpy as np
import argparse
import os
import glob

def stitch_images_from_folder(folder_path, superpoint_model_path="superpoint.onnx", lightglue_model_path="lightglue.onnx"):
    """
    从指定文件夹读取图像并进行全景拼接
    
    参数:
        folder_path: 包含待拼接图像的文件夹路径
        superpoint_model_path: SuperPoint模型文件路径
        lightglue_model_path: LightGlue模型文件路径
    """
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹 '{folder_path}' 不存在")
        return False
    
    # 从文件夹读取图像
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    images = []
    
    for extension in image_extensions:
        image_paths = glob.glob(os.path.join(folder_path, extension))
        image_paths.extend(glob.glob(os.path.join(folder_path, extension.upper())))
        
        for image_path in image_paths:
            img = cv2.imread(image_path)
            if img is not None:
                images.append(img)
                print(f"已加载: {os.path.basename(image_path)}")
            else:
                print(f"警告: 无法读取图像 {image_path}")
    
    # 检查是否找到足够图像
    if len(images) < 2:
        print(f"错误: 需要至少2张图像进行拼接，当前找到 {len(images)} 张")
        return False
    
    print(f"找到 {len(images)} 张图像，开始拼接...")
    
    # 创建 SuperPoint 特征检测器
    try:
        superpoint = cv2.SuperPoint.create(superpoint_model_path)
        print("SuperPoint 特征检测器创建成功")
    except Exception as e:
        print(f"SuperPoint 创建失败: {e}")
        print("使用 ORB 作为特征检测器")
        superpoint = cv2.ORB.create()
    
    # 创建拼接器并设置特征检测器
    stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
    stitcher.setFeaturesFinder(superpoint)
    
    # 尝试设置 LightGlue 匹配器
    try:
        lightglue = cv2.detail.LightGlue.create(lightglue_model_path)
        stitcher.setFeaturesMatcher(lightglue)
        print("LightGlue 匹配器设置成功")
    except Exception as e:
        print(f"LightGlue 设置失败: {e}")
        print("使用默认匹配器")
    
    # 执行拼接
    print("开始图像拼接过程，这可能需要一些时间...")
    status, pano = stitcher.stitch(images)
    
    # 处理拼接结果
    if status == cv2.Stitcher_OK:
        # 创建输出文件名
        output_path = os.path.join(folder_path, "panorama_result.jpg")
        
        # 保存全景图
        cv2.imwrite(output_path, pano)
        print(f"高级全景拼接成功！")
        print(f"全景图像已保存至: {output_path}")
        print(f"全景图像尺寸: {pano.shape[1]}x{pano.shape[0]} (宽x高)")
        
        # 显示结果（可选）
        cv2.imshow("Panorama Result", pano)
        print("按任意键关闭显示窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return True
    else:
        error_messages = {
            cv2.Stitcher_ERR_NEED_MORE_IMGS: "需要更多图像进行拼接",
            cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "单应性矩阵估计失败",
            cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "相机参数调整失败"
        }
        error_msg = error_messages.get(status, f"未知错误，代码: {status}")
        print(f"拼接失败: {error_msg}")
        return False

if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='全景图像拼接工具')
    parser.add_argument('--folder_path', help='包含待拼接图像的文件夹路径',default='./test_imgs')
    parser.add_argument('--superpoint', default='/root/weights/superpoint.onnx', 
                       help='SuperPoint模型文件路径 (默认: superpoint.onnx)')
    parser.add_argument('--lightglue', default='/root/weights/superpoint_lightglue.onnx', 
                       help='LightGlue模型文件路径 (默认: lightglue.onnx)')
    
    args = parser.parse_args()
    
    # 执行拼接
    success = stitch_images_from_folder(
        args.folder_path, 
        args.superpoint, 
        args.lightglue
    )
    
    if success:
        print("程序执行完成")
    else:
        print("程序执行失败")
        exit(1)