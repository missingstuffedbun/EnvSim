import gymnasium as gym
import numpy as np


class FloodEnv(gym.Env):
    def __init__(
            self,
            seed: int = 2024,
            max_steps: int = 1000,
            **kwargs
    ):
        super(FloodEnv, self).__init__()

    # def _val_init_params(self, *args):
    #     # 如果所有参数都为空，抛出异常
    #     if all(param is None for param in args):
    #         raise ValueError("至少一个参数必须提供。")
    #
    #     # 获取现有数组的形状
    #     provided_shapes = [param.shape for param in args if param is not None]
    #     if len(set(provided_shapes)) > 1:
    #         raise ValueError("所有提供的数组必须具有相同的形状。")
    #
    #     # 确定目标形状
    #     shape = provided_shapes[0] if provided_shapes else (0, 0)
    #     self.shape = shape
    #
    #     for arg in args:
    #         setattr(self, f'_{arg.__name__}', arg)

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



