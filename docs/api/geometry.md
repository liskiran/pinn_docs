# Geometry

## BaseGeometry
    class BaseGeometry(ABC)

Абстрактный класс для наследования классов, отвеающих за геометрию

**Методы**

 - **generate_points(self, condition, model)** : Абстрактный метод для генерации точек внутри области


## RectangleArea
    class RectangleArea(self, low, high, generator: BaseGenerator = None)

Класс геометрий, определяющий прямоугольные области

**Параметры** 

- **low** (np.array) : массив со значениями нижних границ для каждой оси
- **high** (np.array) : массив со значениями верхних границ для каждой оси
- **generator** (BaseGeerator) : объект генератора, определяющий алгоритм генерации точек в области

**Методы** 

- **generate_points** (condition, model) -> (torch.tensor, torch.tensor) : функция генерации точек, лежащих в заданных границах области

**Пример**

```python
inlet_reg = RectangleArea(low=[0.0, 0.0], high=[5.0, 1.0])
middle_reg = RectangleArea(low=[4.0, 1.0], high=[5.0, 2.0])
outlet_reg = RectangleArea(low=[4.0, 2.0], high=[9.0, 3.0])
```

## MeshStorage
    class MeshStorage(self, points: torch.Tensor, generator:BaseGenerator = None)

Класс, позволяющий добавлять в обучение новое множество точек, не представимое в виде объекта типа Face. Данный класс используется
в реализации эксперимена с добавлением информации о точном решении.

**Параметры** 

- **points** (torch.tensor) : точки  используемые для обучения
- **generator** (BaseGenerator) : генератор определяющий параметры и алгоритм генерации точек

**Методы**

- **generate_points** (condition, model) -> (torch.tensor, torch.tensor) : функция генерации точек

**Пример**

```python
mesh_sol = meshio.read(solution_file_path)  # считываем точки из файла
points = mesh_sol.points
slice_range = [0.5, 0.5 + 0.05]
indices = (points[:, 2] >= slice_range[0]) & (points[:, 2] <= slice_range[1])  # фильтруем точки, берем только те, которые лежат в нужном диапазоне
points = torch.tensor(points[indices1].astype(np.float32))
slice_points = MeshStorage(points)  # создаем объект MeshStorage
```




## MeshArea
    class MeshArea(self, face: Face, n_dims: int, generator: BaseGenerator = None)

Класс, позволяющий добавлять в обучение новое множество точек, хранящееся в mesh файле. Используется для работы с областями заданными множеством точек.

**Параметры** 


- **n_connections** (torch.tensor) : количество точек
- **n_dims** (torch.tensor) : размерность области
- **points** (torch.tensor) :  точки  используемые для обучения
- **normals** (torch.tensor) :  нормали в точках для обучения. Присутствуют только в граничных точках
- **generator** (BaseGenerator) : генератор определяющий параметры и алгоритм генерации точек

**Методы**

- **generate_points** (condition, model) -> (torch.tensor, torch.tensor) : функция генерации точек

**Пример**

```python

mesh_grid = GridReader().read(mesh_file_path)  # Считываем файл. Может использоваться файл с .pt расширением
zone_names = mesh_grid.zones_names  # достаём названия областей

inner_dom = MeshArea(mesh_grid.get_face_by_id(zone_names['inner_zone']), mesh_grid.dim)
walls_dom = MeshArea(mesh_grid.get_face_by_id(zone_names['wall_zone']), mesh_grid.dim)
inlet_dom = MeshArea(mesh_grid.get_face_by_id(zone_names['inlet_zone']), mesh_grid.dim)
outlet_dom = MeshArea(mesh_grid.get_face_by_id(zone_names['outlet_zone']), mesh_grid.dim) # cоздаем объекты областей
```

