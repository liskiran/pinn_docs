# \_unpack

```python
def _unpack(batch: torch.Tensor) -> torch.Tensor:

```

Возвращает распакованные аргументы тензора

## Параметры

- ** batch ** - Входной тензор

## Пример:

```python
x, y, z = _unpack(arg) the same as x, y, z = arg[:, 0], arg[:, 1], arg[:, 2] # распаковываем входные аргументы
u, v, w = _unpack(f) # распаковываем выходные аргументы
u_xx, u_xy, u_xz = _unpack(_grad(u_x, arg)) # распаковываем градиент от функции
u_y, v_y, w_y = _unpack(_num_diff(arg, model, f, [0, 1, 0])) # распаковываем численные производные от функции
```

---

# \_grad

```python
def _grad(u: torch.Tensor, arg: torch.Tensor) -> torch.Tensor:
```

Возвращает градиент от функции

## Параметры

- ** u ** - Функция, от которой берут производную
- ** arg ** - Аргументы, по которым берут производные

---

# \_num_diff

```python
def _num_diff(arg, model, f, direction, eps=1E-4) -> torch.Tensor:
```

Возвращает численную аппроксимацию производной

## Параметры

- ** arg ** - аргумент, по которому дифференцируем
- ** model ** - модель нейронной сети
- ** f ** - функция аппроксимации
- ** direction ** - направление
- ** eps ** - порядок аппроксимации

---

# \_num_diff_random

```python
def _num_diff_random(arg, model, f, direction, max_eps=1E-3, min_eps=1E-6) -> torch.Tensor:
```

Возвращает производную в рандомной аппроксимации

## Параметры

- ** arg ** - аргумент, по которому дифференцируем
- ** model ** - модель нейронной сети
- ** f ** - фунцкции аппроксимации
- ** direction ** - направление
- ** max_eps ** - максимальная граница аппроксимации
- ** min_eps ** - минимальная граница аппроксимации

---

# \_random_spherical

```python
def _random_spherical(ndim: int) -> torch.Tensor:
```

Возвращает векторную норму

- ** ndim ** - размерность

---

# \_diff_residual

```python
def _diff_residual(arg, model, eps=1E-4) -> torch.Tensor:
```

Возвращает производную от невязки

## Параметры

- ** arg ** - аргумент, передающую в модель
- ** model ** - модель
- ** eps ** - граница аппроксимации

---

# problem_2D1C_heat_equation

Уравнение теплопроводности с двумя граничными условиями

$$ u*t + u*{xx} = e^{-\gamma^2t} \cdot (-\gamma^2 \cdot (a \cdot \cos{\alpha x} + b \cdot \sin{beta \cdot x}) + b \cdot \beta^2 \* \sin{beta \cdot x} + a \cdot \alpha^2 \cdot \cos{\alpha \cdot x})$$

$$ u = a \cdot \cos{\alpha \cdot x} + b \cdot \sin{\beta \cdot x} $$

$$ u = e^{-\gamma^2 \cdot t} \cdot a$$

$$ u = e^{\gamma^{2 \cdot t}} \cdot a \cdot \cos{\alpha \cdot 2} + b \cdot \sin{\beta \cdot 2}$$

---

# first_problem_2D1C

Уравнение теплопроводности

$$ u*t + u*{xx} = 2 - 4 \cdot e^{2x}, x \in [0, 2], y \in [0, 1] $$

$$ \left. u \right|\_{t = 0} = e^{2x} $$

$$ \left. u \right|\_{x = 0} = 2t + 1 $$

$$ \left. u \right|\_{x = 2} = 2t + e^4 $$

---

# problem_2D2C_heat_equation

Уравнение теплопроводности с одним граничным условием

$$
\begin{equation}
    \begin{cases}
    u_t - u_{xx} = e^{\pi x} \\
    v_x = u
 \end{cases}\
\end{equation}, \ x \in [0, 1], y \in [0, 1]
$$

$$ \left. u \right|\_{t = 0} = \cos{\pi x} $$

$$
\begin{equation}
    \begin{cases}
    \left. u \right|_{x = 0} = e^{-t \cdot \pi^2} \\
    \left. v \right|_{x = 0} = t
    \end{cases}\
\end{equation}
$$

$$ \left. u*x \right|*{x = 1} = 0$$

# TODO

- ** problem_3D1C_laplace ** - Уравнение Лапласа
- ** problem_1D1C_portal ** - # TODO
- ** problem_2D1C_Allen_Cahn ** - Уравнение Аллан-Кана

---

# Как написать свое уравнение:

## 1. Задайте внутренние и граничные условия

```python
 def ic(arg, model):
     f, u, x, t = basic_symbols(arg, model)
     assert torch.all(torch.isclose(t, torch.zeros_like(t)))
     return [u - a * torch.cos(alpha * x) - b * torch.sin(beta * x)]

 def bc1(arg, model):
     f, u, x, t = basic_symbols(arg, model)
     assert torch.all(torch.isclose(x, torch.zeros_like(x)))
     return [u - torch.exp(-gamma ** 2 * t) * a]

 def bc2(arg, model):
     f, u, x, t = basic_symbols(arg, model)
     assert torch.all(torch.isclose(x, torch.ones_like(x) * 2))
     return [u - torch.exp(-gamma ** 2 * t) * (
                 a * torch.cos(torch.tensor(alpha * 2)) + b * torch.sin(torch.tensor(beta * 2)))]
```

## 2. Задайте внутренние и граничные области

```python
domain = RectangleArea(low=[0, 0], high=[2, 2])
x_min = RectangleArea(low=[0, 0], high=[0, 2])
x_max = RectangleArea(low=[2, 0], high=[2, 2])
t_0 = RectangleArea(low=[0, 0], high=[2, 0])
```

## 3. Поместите внутренние и граничные условия в класс Condition:

```python
pde = [ # Связываем каждую область с граничными или внутреннеми областями
    Condition(inner, domain),
    Condition(bc1, x_min),
    Condition(bc2, x_max),
    Condition(ic, t_0),
]
```

#### Пример

```python
def problem_2D1C_heat_equation(a=1, b=0.5, alpha=0.5, beta=10, gamma=0.7):
    input_dim = 2 # размерность входных данных (x, t) в примере
    output_dim = 1 # размерность выходных данных (u) в примере, может быть больше, если решаем систему уравнений

    def basic_symbols(arg, model): # функция, для удобной распаковки переменных
        f = model(arg) # получаем f
        u, = _unpack(f) # получаем u
        x, t = _unpack(arg) # получаем входные данные в виде (x, t)
        return f, u, x, t # возвращаем переменные в виде кортежа

    def inner(arg, model): # уравнение во внутренней области
        f, u, x, t = basic_symbols(arg, model) # распаковка переменных
        u_x, u_t = _unpack(_grad(u, arg)) # получаем первые производные от функции
        u_xx, u_xt = _unpack(_grad(u_x, arg)) # получаем вторые производные от функции
        eq1 = u_t - u_xx - torch.exp(-gamma ** 2 * t) * (
                    -gamma ** 2 * (a * torch.cos(alpha * x) + b * torch.sin(beta * x)) + b * beta ** 2 * torch.sin(
                beta * x) + a * alpha ** 2 * torch.cos(alpha * x)) # записываем наше уравнение
        return [eq1] # возвращаем в виде списка уравнения (если есть система уравнение, то возвращаем [eq1, eq2, ..., eqn])

    # Задаем внутренние и граничные условия:
    def ic(arg, model):
        f, u, x, t = basic_symbols(arg, model)
        assert torch.all(torch.isclose(t, torch.zeros_like(t)))
        return [u - a * torch.cos(alpha * x) - b * torch.sin(beta * x)]

    def bc1(arg, model):
        f, u, x, t = basic_symbols(arg, model)
        assert torch.all(torch.isclose(x, torch.zeros_like(x)))
        return [u - torch.exp(-gamma ** 2 * t) * a]

    def bc2(arg, model):
        f, u, x, t = basic_symbols(arg, model)
        assert torch.all(torch.isclose(x, torch.ones_like(x) * 2))
        return [u - torch.exp(-gamma ** 2 * t) * (
                    a * torch.cos(torch.tensor(alpha * 2)) + b * torch.sin(torch.tensor(beta * 2)))]

    # Задаем области
    # low = [x1_min, x2_min, ..., xn_min]
    # high = [y1_max, y2_max, ..., yn_max]

    domain = RectangleArea(low=[0, 0], high=[2, 2])
    x_min = RectangleArea(low=[0, 0], high=[0, 2])
    x_max = RectangleArea(low=[2, 0], high=[2, 2])
    t_0 = RectangleArea(low=[0, 0], high=[2, 0])

    pde = [ # Помещаем условия с областями в виде Condition
        Condition(inner, domain),
        Condition(bc1, x_min),
        Condition(bc2, x_max),
        Condition(ic, t_0),
    ]

    return pde, input_dim, output_dim # возвращаем кортеж из (условии, входная размерность, выходная размерность)
```
