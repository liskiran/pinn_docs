# Задача

В библиотеке реализованы многие физические и математические задачи различной размерности:
- теплопроводности
- Лапласса 
- Алана-Кана
- Пуассона
- Навье-Стокса
- 
## unpack

```python
def _unpack(batch: torch.Tensor) -> torch.Tensor:
```

Возвращает распакованные аргументы тензора. (Tensor)

** Параметры **

- ** batch ** (Tensor) - Входной тензор.

** Пример **

```python
x, y, z = _unpack(arg) # (1)!
u, v, w = _unpack(f) # (2)!
u_xx, u_xy, u_xz = _unpack(_grad(u_x, arg)) # (3)!
u_y, v_y, w_y = _unpack(_num_diff(arg, model, f, [0, 1, 0])) # (4)!
```

1. Распаковываем входные аргументы.
2. Распаковываем выходные аргументы.
3. Распаковываем градиент от функции.
   В данном примере функция `u_x` раскалыдывается на три вектора `u_xx, u_xy, u_xz`
4. Распаковываем численные производные от функции.
   В данном примере функцию `u_x` раскалыдывается на три вектора
   `u_y, v_y, w_y`, которые расскладывается по одному аргументу
   по всем выходным векторам.

---

## grad

```python
def _grad(u: torch.Tensor, arg: torch.Tensor) -> torch.Tensor:
```

Возвращает градиент от функции. (Tensor)

**Параметры **

- ** u ** (Tensor) - Функция, от которой берут производную.
- ** arg ** (Tensor) - Аргументы, по которым берут производные.

**Пример**

```python
u_xx, u_xy, u_xz = _unpack(_grad(u_x, arg)) # (1)!
```

1. Возвращает тензор градиентов по всем аргументам. В данном примере ищем градиент `u_x` и получаем на выходе: `u_xx, u_xy, u_xz`.

---

## num_diff

```python
def _num_diff(arg, model, f, direction, eps=1e-4) -> torch.Tensor:
```

Возвращает численную аппроксимацию производной. (Tensor)

** Параметры**

- ** arg ** - аргумент, по которому дифференцируем.
- ** model ** - модель нейронной сети.
- ** f ** - функция аппроксимации.
- ** direction ** - направление.
- ** eps ** - порядок аппроксимации.

---

## num_diff_random

```python
def _num_diff_random(arg, model, f, direction, max_eps=1e-3, min_eps=1e-6) -> torch.Tensor:
```

Возвращает производную в случайной аппроксимации. (Tensor)

** Параметры **

- ** arg ** - аргумент, по которому дифференцируем.
- ** model ** - модель нейронной сети.
- ** f ** - фунцкции аппроксимации.
- ** direction ** - направление.
- ** max_eps ** - максимальная граница аппроксимации.
- ** min_eps ** - минимальная граница аппроксимации.

---

## random_spherical

```python
def _random_spherical(ndim: int) -> torch.Tensor:
```

Возвращает векторную норму. (Tensor)

- ** ndim ** - размерность.

---

## diff_residual

```python
def _diff_residual(arg, model, eps=1e-4) -> torch.Tensor:
```

Возвращает производную от невязки.

** Параметры **

- ** arg ** - аргумент, передающую в модель.
- ** model ** - модель нейронной сети.
- ** eps ** - граница аппроксимации.

---

## problem_2D1C_heat_equation

```python
def problem_2D1C_heat_equation(a=1, b=0.5, alpha=0.5, beta=10, gamma=0.7):
```

Уравнение теплопроводности с двумя граничными условиями.

```python
def inner(arg, model):
    f, u, x, t = basic_symbols(arg, model)
    u_x, u_t = _unpack(_grad(u, arg))
    u_xx, u_xt = _unpack(_grad(u_x, arg))
    eq1 = u_t - u_xx - torch.exp(-gamma ** 2 * t) * (
                -gamma ** 2 * (a * torch.cos(alpha * x) + b * torch.sin(beta * x)) + b * beta ** 2 * torch.sin(
            beta * x) + a * alpha ** 2 * torch.cos(alpha * x))
    return [eq1]
```

$$
u_t + u_{xx} = e^{-\gamma^2t}  (-\gamma^2 (a cos(\alpha x) + \
b sin(\beta x)) + b  \beta^2 sin(beta x) + a \alpha^2 cos(\alpha x))
$$

```python
def ic(arg, model):
    f, u, x, t = basic_symbols(arg, model)
    assert torch.all(torch.isclose(t, torch.zeros_like(t)))
    return [u - a * torch.cos(alpha * x) - b * torch.sin(beta * x)]
```

$$ u = a cos(\alpha x) + b sin(\beta x) $$

```python
def bc1(arg, model):
    f, u, x, t = basic_symbols(arg, model)
    assert torch.all(torch.isclose(x, torch.zeros_like(x)))
    return [u - torch.exp(-gamma ** 2 * t) * a]
```

$$ u = e^{-\gamma^2 t} a$$

```python
def bc2(arg, model):
    f, u, x, t = basic_symbols(arg, model)
    assert torch.all(torch.isclose(x, torch.ones_like(x) * 2))
    return [u - torch.exp(-gamma ** 2 * t) * (
                a * torch.cos(torch.tensor(alpha * 2)) + b * torch.sin(torch.tensor(beta * 2)))]

```

$$ u = e^{\gamma^{2 t}} a cos(2 \alpha) + b sin(2\beta) $$

---

## first_problem_2D1C

```python
def first_problem_2D1C() -> None:
```

Уравнение теплопроводности.

```python
def inner(arg, model):
    f, u, x, t = basic_symbols(arg, model)
    u_x, u_t = _unpack(_grad(u, arg))
    u_xx, u_xt = _unpack(_grad(u_x, arg))
    eq1 = u_t - u_xx - (2 - 4 * torch.exp(2 * x))
    return [eq1]
```

$$
u_t + u_{xx} = 2 - 4  e^{2x}, x \in [0, 2], y \in [0, 1]
$$

```python
def ic(arg, model):
    f, u, x, t = basic_symbols(arg, model)
    assert torch.all(torch.isclose(t, torch.zeros_like(t)))
    return [torch.exp(2 * x) - u]
```

$$
\left. u \right|_{t = 0} = e^{2x}
$$

```python
def bc1(arg, model):
    f, u, x, t = basic_symbols(arg, model)
    assert torch.all(torch.isclose(x, torch.zeros_like(x)))
    return [1 + 2 * t - u]
```

$$
\left. u \right|_{x = 0} = 2t + 1
$$

```python
def bc2(arg, model):
    f, u, x, t = basic_symbols(arg, model)
    assert torch.all(torch.isclose(x, 2 * torch.ones_like(x)))
    return [torch.exp(torch.tensor(4)) + 2 * t - u]
```

$$
\left. u \right|_{x = 2} = 2t + e^4
$$

---

## problem_2D2C_heat_equation

```python
def problem_2D2C_heat_equation():
```

Уравнение теплопроводности с одним граничным условием.

```python
def inner(arg, model):
    t, x, u, u_x, u_t, u_xx, v, v_x = unpack_symbols(arg, model)
    eq1 = u_t - u_xx - torch.exp(x)
    eq2 = v_x - u  # v == integral u of dx
    return [eq1, eq2]
```

$$
\begin{equation}
    \begin{cases}
    u_t - u_{xx} = e^{x} \\
    v_x = u
 \end{cases}\
\end{equation}, \ x \in [0, 1], y \in [0, 1]
$$

```python
def ic(arg, model):
    t, x, u, u_x, u_t, u_xx, v, v_x = unpack_symbols(arg, model)
    assert torch.all(torch.isclose(t, torch.zeros_like(t)))
    return [u - torch.cos(torch.pi * x)]
```

$$
\left. u \right|_{t = 0} = cos(\pi x)
$$

```python
def bc1(arg, model):
    t, x, u, u_x, u_t, u_xx, v, v_x = unpack_symbols(arg, model)
    assert torch.all(torch.isclose(x, torch.zeros_like(x)))
    return [u - torch.exp(-t * torch.pi ** 2), v - t]
```

$$
\begin{equation}
    \begin{cases}
    \left. u \right|_{x = 0} = e^{-\pi^2 t} \\
    \left. v \right|_{x = 0} = t
    \end{cases}\
\end{equation}
$$

```python
def bc2(arg, model):
    t, x, u, u_x, u_t, u_xx, v, v_x = unpack_symbols(arg, model)
    assert torch.all(torch.isclose(x, torch.ones_like(x)))
    return [u_x]
```

$$
\left. u_x \right|_{x = 1} = 0
$$

---

## Как написать свое уравнение:

1. Задайте внутренние и граничные условия:

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

2. Задайте внутренние и граничные области:

```python
domain = RectangleArea(low=[0, 0], high=[2, 2])
x_min = RectangleArea(low=[0, 0], high=[0, 2])
x_max = RectangleArea(low=[2, 0], high=[2, 2])
t_0 = RectangleArea(low=[0, 0], high=[2, 0])
```

3. Поместите внутренние и граничные условия в класс Condition:

```python
pde = [ # Связываем каждую область с граничными или внутреннеми областями
    Condition(inner, domain),
    Condition(bc1, x_min),
    Condition(bc2, x_max),
    Condition(ic, t_0),
]
```

** Пример **

```python
def problem_2D1C_heat_equation(a=1, b=0.5, alpha=0.5, beta=10, gamma=0.7):
    input_dim = 2 # (1)!
    output_dim = 1 # (2)!

    def basic_symbols(arg, model): # (3)!
        f = model(arg) # (4)!
        u, = _unpack(f) # (5)!
        x, t = _unpack(arg) # (6)!
        return f, u, x, t # (7)!

    def inner(arg, model): # (8)!
        f, u, x, t = basic_symbols(arg, model) # (9)!
        u_x, u_t = _unpack(_grad(u, arg)) # (10)!
        u_xx, u_xt = _unpack(_grad(u_x, arg)) # (11)!
        eq1 = u_t - u_xx - torch.exp(-gamma ** 2 * t) * (
                    -gamma ** 2 * (a * torch.cos(alpha * x) + b * torch.sin(beta * x)) + b * beta ** 2 * torch.sin(
                beta * x) + a * alpha ** 2 * torch.cos(alpha * x)) # (12)!
        return [eq1] # (13)!

    # (14)!
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

    # (15)!
    domain = RectangleArea(low=[0, 0], high=[2, 2])
    x_min = RectangleArea(low=[0, 0], high=[0, 2])
    x_max = RectangleArea(low=[2, 0], high=[2, 2])
    t_0 = RectangleArea(low=[0, 0], high=[2, 0])

    pde = [ # (16)!
        Condition(inner, domain),
        Condition(bc1, x_min),
        Condition(bc2, x_max),
        Condition(ic, t_0),
    ]

    return pde, input_dim, output_dim # (17)!
```

1. Размерность входных данных (x, t) в примере.
2. размерность выходных данных (u) в примере, может быть больше, если решаем систему уравнений.
3. Функция, для удобной распаковки переменных.
4. Получаем f
5. Получаем u
6. Получаем входные данные в виде (x, t)
7. Возвращаем переменные в виде кортежа
8. Уравнение во внутренней области
9. Распаковка переменных
10. Получаем первые производные от функции
11. Получаем вторые производные от функции
12. Записываем наше уравнение
13. Возвращаем в виде списка уравнения (если есть система уравнение, то возвращаем [eq1, eq2, ..., eqn])
14. Задаем внутренние и граничные условия:
15. Задаем области: low = [x1_min, x2_min, ..., xn_min]; high = [y1_max, y2_max, ..., yn_max]
16. Помещаем условия с областями в виде Condition
17. Возвращаем кортеж из (условии, входная размерность, выходная размерность)
