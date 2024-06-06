import gymnasium as gym
import numpy as np

from scipy import ndimage

from Env.data_loader import scale_image


def mask_safe_region(shelters, region_size):
    # 创建一个与 shelters 相同大小的数组，用于存储安全区域信息
    shelters_safe = shelters.copy()

    # 进行膨胀操作
    struct = ndimage.generate_binary_structure(2, 2)
    shelters_safe = ndimage.binary_dilation(shelters_safe, structure=struct, iterations=region_size).astype(shelters_safe.dtype)

    # print("Num of shelters: {}    Safe area: {}".format(np.count_nonzero(shelters), np.count_nonzero(shelters_safe)))
    # plt.imshow(shelters_safe, cmap='binary')  # 0为白色，1为黑色
    # plt.show()

    return shelters_safe

def build_shelters(
        dem,
        num_top_buildings: int = 10,
        region_size: int = 100):

    shelters = np.zeros_like(dem)

    # 找到 DEM 数值最高的位置索引
    max_dem_indices = np.unravel_index(np.argsort(dem, axis=None)[-num_top_buildings:], dem.shape)
    # 在 shelters 数组中将这些位置标记为 1
    shelters[max_dem_indices] = 1

    # 相邻区域范围
    shelters_safe = mask_safe_region(shelters, region_size)

    while not np.all(shelters_safe == 1):
        # 找到 shelters_safe 中值为 False 的位置
        false_indices = np.argwhere(shelters_safe == False)

        # 初始化最高 DEM 值和位置
        max_dem_value = -np.inf
        max_dem_position = None

        # 对于每个 False 的位置，找到 DEM 最高的位置
        for idx in false_indices:
            # 获取当前位置的 DEM 值
            dem_value = dem[idx[0], idx[1]]
            # 如果当前 DEM 值大于最高 DEM 值，则更新最高 DEM 值和位置
            if dem_value > max_dem_value:
                max_dem_value = dem_value
                max_dem_position = idx

        # 将 max_dem_position 位置标记为 1
        shelters[max_dem_position[0], max_dem_position[1]] = 1

        shelters_safe = mask_safe_region(shelters, region_size)

        # 进行膨胀操作
        struct = ndimage.generate_binary_structure(2, 2)
        shelters = ndimage.binary_dilation(shelters, structure=struct, iterations=30).astype(shelters.dtype)

        return shelters


class FloodEnv(gym.Env):
    def __init__(
            self,
            seed: int = 2024,
            max_steps: int = 1000,
            **kwargs
    ):
        super(FloodEnv, self).__init__()

        self.simplification(0.05)

    @property
    def buildings(self):
        return self._buildings

    @buildings.setter
    def buildings(self, value: np.ndarray[int]):
        self._buildings = value

    @property
    def roads(self):
        return self._roads

    @roads.setter
    def roads(self, value: np.ndarray[int]):
        self._roads = value

    @property
    def shelters(self):
        return self._shelters

    @shelters.setter
    def shelters(self, value: np.ndarray[int]):
        self._shelters = value

    @property
    def floods(self):
        return self._floods

    @floods.setter
    def floods(self, value: np.ndarray[float]):
        self._floods = value

    @property
    def population(self):
        return self._population

    @population.setter
    def population(self, value: np.ndarray[int]):
        self._population = value

    @property
    def dem(self):
        return self._dem

    @dem.setter
    def dem(self, value: np.ndarray[float]):
        self._dem = value

    @property
    def diffusion_coefs(self):
        return self._diffusion_coefs

    @diffusion_coefs.setter
    def diffusion_coefs(self, value: np.ndarray[float]):
        self._diffusion_coefs = value

    def simplification(self, factor=0.1):
        buildings = scale_image(self._buildings, factor)
        self._buildings = buildings.astype(int)

        dem = scale_image(self._dem, factor)
        self._dem = dem.astype(float)

        roads = scale_image(self._roads, factor)
        self._roads = roads.astype(int)

        floods = scale_image(self._floods, factor)
        self._floods = floods.astype(float)

        shelters = scale_image(self._shelters, factor)
        self._shelters = shelters.astype(int)

        population = scale_image(self._population, factor)
        self._population = population.astype(int)

        diffusion_coefs = scale_image(self._diffusion_coefs, factor)
        self._diffusion_coefs = diffusion_coefs.astype(float)







