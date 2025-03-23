import numpy as np
import cv2
import os
import xml.etree.ElementTree as ET
from scipy.interpolate import RegularGridInterpolator


def load_lut(file_path):
    """ 自适应加载 .cube, .look, .3dl LUT 文件 """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".cube":
        return load_cube_lut(file_path)
    elif ext == ".look":
        return load_look_lut(file_path)
    elif ext == ".3dl":
        return load_3dl_lut(file_path)
    else:
        raise ValueError(f"Unsupported LUT format: {ext}")


def load_cube_lut(file_path):
    """ 解析 .cube 格式 """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    lut_data = []
    size = None

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()

        if parts[0].lower() == "lut_3d_size":
            size = int(parts[-1])
            continue
        if any(c.isalpha() for c in parts[0]):
            continue

        values = list(map(float, parts))
        if len(values) == 3:
            lut_data.append(values)

    if size is None or len(lut_data) != size ** 3:
        raise ValueError("Invalid .cube file format.")

    return np.array(lut_data, dtype=np.float32).reshape((size, size, size, 3))


import numpy as np
import xml.etree.ElementTree as ET

def load_look_lut(file_path):
    """ 解析 .look LUT 文件（没有 ColorCorrect） """
    tree = ET.parse(file_path)
    root = tree.getroot()

    # 直接读取 `<Look>` 里的数据
    slope = np.array(list(map(float, root.find("Slope").text.split())), dtype=np.float32)
    offset = np.array(list(map(float, root.find("Offset").text.split())), dtype=np.float32)
    power = np.array(list(map(float, root.find("Power").text.split())), dtype=np.float32)
    saturation = float(root.find("Saturation").text)

    # 生成 LUT
    size = 33
    lut = np.zeros((size, size, size, 3), dtype=np.float32)
    grid = np.linspace(0, 1, size)

    for i, r in enumerate(grid):
        for j, g in enumerate(grid):
            for k, b in enumerate(grid):
                rgb = np.array([r, g, b], dtype=np.float32)
                rgb = (rgb * slope + offset) ** power
                lum = np.dot(rgb, [0.2126, 0.7152, 0.0722])
                rgb = lum + saturation * (rgb - lum)
                lut[i, j, k] = np.clip(rgb, 0, 1)

    return lut



import numpy as np

def load_3dl_lut(file_path):
    """ 解析 .3dl LUT 文件，支持有无 `3DLUTSIZE` 关键字 """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    lut_data = []
    size = None

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue  # 跳过空行和注释行
        parts = line.split()
        
        if len(parts) == 2 and parts[0].lower() == "3dlutsize":
            size = int(parts[1])  # 提取 LUT 尺寸
            print(f"Detected LUT size: {size}")  # 调试输出
            continue

        if any(c.isalpha() for c in parts[0]):  
            continue  # 跳过包含字母的行

        try:
            values = list(map(float, parts))
            if len(values) == 3:
                lut_data.append(values)
        except ValueError as e:
            print(f"Error parsing line: {parts} -> {e}")

    # 自动检测 LUT 尺寸（如果未提供 `3DLUTSIZE`）
    if size is None:
        estimated_size = round(len(lut_data) ** (1/3))
        if estimated_size**3 == len(lut_data):
            size = estimated_size
            print(f"Auto-detected LUT size: {size}")
        else:
            raise ValueError("Cannot determine LUT size. Please check the .3dl file format.")

    # 检查 LUT 数据长度
    if len(lut_data) != size ** 3:
        raise ValueError(f"Invalid .3dl file format. Expected {size**3} entries, found {len(lut_data)}")

    print(f"LUT loaded successfully: {size}x{size}x{size}")

    return np.array(lut_data, dtype=np.float32).reshape((size, size, size, 3))



def apply_3d_lut(image, lut):
    """ 应用 3D LUT 到图像 """
    h, w, c = image.shape
    assert c == 3, "Image must have 3 channels (RGB)"

    img_norm = image.astype(np.float32) / 255.0
    size = lut.shape[0]
    grid = np.linspace(0, 1, size)

    interp_func = RegularGridInterpolator((grid, grid, grid), lut, bounds_error=False, fill_value=None)
    img_reshaped = img_norm.reshape(-1, 3)
    img_transformed = interp_func(img_reshaped).reshape(h, w, 3)

    return np.clip(img_transformed * 255, 0, 255).astype(np.uint8)


def generate_output_path(image_path, lut_path):
    """ 生成输出图片路径 """
    img_name, img_ext = os.path.splitext(os.path.basename(image_path))
    lut_name = os.path.splitext(os.path.basename(lut_path))[0]
    return f"{img_name}_{lut_name}{img_ext}"


def main():
    image_path = r"D:\workspace\pycharm\color\img\1.jpg"
    lut_path = r"D:\workspace\pycharm\color\luts\Duo-toning.look"
    output_path=image_path.replace(r'color\img',r'color\result')


    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError("Image not found.")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    lut = load_lut(lut_path)

    output_image_rgb = apply_3d_lut(image_rgb, lut)
    # output_path = generate_output_path(image_path, lut_path)
    output_imname = generate_output_path(image_path, lut_path)



    output_path=os.path.join(os.path.dirname(output_path),output_imname)


    # output_image_bgr = cv2.cvtColor(output_image_rgb, cv2.COLOR_RGB2BGR)


    output_image_bgr=np.array(output_image_rgb)
    cv2.imwrite(output_path, output_image_bgr)

    print(f"Processed image saved as {output_path}")



if __name__ == "__main__":
    main()
