import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def tif_to_array(file_path: str, show=False, save=False, name="array"):
    """
    将TIFF图像文件转换为numpy数组。

    参数:
    file_path: 字符串，TIFF图像文件的路径。
    show: 布尔值，是否显示图像，默认为False。
    save: 布尔值，是否保存数组为.npy文件，默认为False。
    name: 字符串，保存数组的文件名，如果不提供，则使用默认名称。

    返回:
    numpy数组，表示TIFF图像的像素值。
    """
    # 尝试打开图像文件
    try:
        img = Image.open(file_path)
    except:
        # 如果文件打开失败，则抛出FileNotFoundError异常
        raise FileNotFoundError(f"文件未找到或无法打开: {file_path}")

    # 将图像对象转换为numpy数组
    arr = np.array(img)

    # 如果设置了显示参数为True，则使用matplotlib显示图像
    if show:
        plt.imshow(arr)

    # 如果设置了保存参数为True，则将数组保存为.npy文件
    if save:
        np.save(f'{name}.npy', arr)
    # 返回转换后的numpy数组
    return arr


def npy_to_array(file_path: str, show=False):
    try:
        arr = np.load(file_path)
    except:
        raise FileNotFoundError(f"文件未找到或无法打开: {file_path}")

    if show:
        plt.imshow(arr)

    return arr