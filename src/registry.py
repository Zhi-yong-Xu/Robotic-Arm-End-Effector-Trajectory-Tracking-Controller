"""通用注册表: type 字符串到类的映射, 支持组件插拔。"""


class Registry:
    """按 key 注册/创建组件的容器。"""

    def __init__(self, name: str):
        self.name = name
        self._items = {}

    def register(self, key: str):
        """装饰器: 将类注册到指定 key, 重复注册抛异常。"""
        def deco(cls):
            if key in self._items:
                raise KeyError(f"[{self.name}] 重复注册: '{key}'")
            self._items[key] = cls
            return cls
        return deco

    def create(self, key: str, *args, **kwargs):
        """按 key 实例化已注册组件, 未注册则抛出可用项列表。"""
        if key not in self._items:
            raise KeyError(f"[{self.name}] 未注册 '{key}', 可用: {self.available}")
        return self._items[key](*args, **kwargs)

    @property
    def available(self):
        return list(self._items)


robot_registry = Registry("robot")
trajectory_registry = Registry("trajectory")
observer_registry = Registry("observer")
kinematic_registry = Registry("kinematic_controller")
dynamic_registry = Registry("dynamic_controller")
