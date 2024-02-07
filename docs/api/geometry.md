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

## MeshStorage
    class MeshStorage(self, points: torch.Tensor, generator:BaseGenerator = None)

Класс, позволяющий добавлять в обучение новое множество точек, не представимое в виде объекта типа Face. Данный класс используется
в реализации эксперимена с добавлением информации о точном решении.

**Параметры** 

- **points** (torch.tensor) : точки  используемые для обучения
- **generator** (BaseGenerator) : генератор определяющий параметры и алгоритм генерации точек

**Методы**

- **generate_points** (condition, model) -> (torch.tensor, torch.tensor) : функция генерации точек



## MeshArea
    class MeshArea(self, face: Face, n_dims: int, generator: BaseGenerator = None)

Класс, позволяющий добавлять в обучение новое множество точек, хранящееся в mesh файле. Используется для работы с областями заданными множеством точек.

**Параметры** 


- **n_points** (torch.tensor) : количество точек
- **n_dims** (torch.tensor) : размерность области
- **points** (torch.tensor) :  точки  используемые для обучения
- **normals** (torch.tensor) :  нормали в точках для обучения. Присутствуют только в граничных точках
- **generator** (BaseGenerator) : генератор определяющий параметры и алгоритм генерации точек

**Методы**

- **generate_points** (condition, model) -> (torch.tensor, torch.tensor) : функция генерации точек

