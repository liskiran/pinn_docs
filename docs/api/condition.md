# Condition

Condition – класс, который содержит в себе информацию о [геометрии](geometry.md) и некоторой заданной функции, которая может являться уравнением, граничным или начальным условием задачи. Один объект отвечает за одно уровнение/условие. 

## Condition
    CLASS src.conditions.condition(function: Callable, geometry: BaseGeometry)


**Параметры**

- **function** (Callable) – функция, заданная как уравнение или начальное/граничное условие. 
- **geometry** (BaseGeometry) – геометрия обалсти, на которой задана функция. 

**Методы**

    update_points(model=None) -> None

Генерирует новый набор точек на геометрии и сохраняет его. В случае, если передается модель, точки генерируются на том же ускорителе (CPU/GPU), на котором находится модель.

    get_residual(model) -> torch.Tensor

Считает функцию от сгенерированного набора точек и модели. **#TODO расписать какие функции должны быть по аргументам.**

**Примеры использования**

```python
input_dim = 2
output_dim = 1

def exact_solution(args):
    return torch.cos(2 * torch.pi * args[:, 0]) * torch.sin(4 * torch.pi * args[:, 1]) + 0.5 * args[:, 0]

def basic_symbols(arg, model):
    f = model(arg)
    u, = _unpack(f)
    x, y = _unpack(arg)
    return f, u, x, y

def inner(arg, model):
    f, u, x, y = basic_symbols(arg, model)
    u_x, u_y = _unpack(_grad(u, arg))
    u_xx, u_xy = _unpack(_grad(u_x, arg))
    u_yx, u_yy = _unpack(_grad(u_y, arg))
    eq1 = u_xx + u_yy + 20 * torch.pi ** 2 * torch.cos(2 * torch.pi * x) * torch.sin(4 * torch.pi * y)
    return [eq1]

domain = RectangleArea(low=[0, 0], high=[0.5, 0.5])

Condition(inner, domain)
```