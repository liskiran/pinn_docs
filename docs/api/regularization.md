# Регуляризация

Регуляризция позволяет быстрее достигать наилучшего решения.

## BigLambdaLossesRegularization

```python
class callbacks.regularization_callback.regularization_callbacks.BigLambdaLossesRegularization(
    ignored_indexes: List[int]
)
```

Класс **BigLambdaLossesRegularization** добавляет при обучении поточечную регуляризацию. Основан на статье
[Investigating and Mitigating Failure Modes in Physicsinformed Neural Networks (PINNs)](https://arxiv.org/abs/2209.09988).

### Параметры:

- **ignored_indexes** (List[int], по умолчанию None) – список индексов уравнений, которые не подвергаются регуляризации.

> **Важно:**
> В данный список следует добавлять индексы уравнений, которые являются основными для вашей задачи. Например, при
> решении задачи движения вязкой несжимаемой жидкости, в данный список следует добавить индексы уравнений Навье-Стокса,
> а уравнение неразрывности, граничные и начальные условия не добавлять.

### Пример использования

```python

def navier_stocks_equation_with_block(Re=50):
    input_dim = 2
    output_dim = 3
    iRe = 1 / Re

    def basic_symbols(arg, model):
        f = model(arg)
        u, v, p = _unpack(f)
        x, y = _unpack(arg)
        return f, u, v, p, x, y

    def inner(arg, model):
        f, u, v, p, x, y = basic_symbols(arg, model)

        u_x, u_y = _unpack(_grad(u, arg))
        v_x, v_y = _unpack(_grad(v, arg))
        p_x, p_y = _unpack(_grad(p, arg))

        u_xx, _ = _unpack(_grad(u_x, arg))
        v_xx, _ = _unpack(_grad(v_x, arg))

        _, u_yy = _unpack(_grad(u_y, arg))
        _, v_yy = _unpack(_grad(v_y, arg))

        laplace_u = u_xx + u_yy
        laplace_v = v_xx + v_yy

        eq1 = u * u_x + v * u_y + p_x - iRe * laplace_u # Уравнениe Навье-Стокса
        eq2 = u * v_x + v * v_y + p_y - iRe * laplace_v # Уравнениe Навье-Стокса
        eq3 = u_x + v_y # Уравнениe неразрывности

        return [eq1, eq2, eq3] # Два первых выхода eq1 и eq2 соответствуют уравнениям Навье Стокса

    def bc_x_min(arg, model):
        f, u, v, p, x, y = basic_symbols(arg, model)
        return [u - (-0.16 * y ** 2 + 1), v]

    def bc_x_max(arg, model):
        f, u, v, p, x, y = basic_symbols(arg, model)
        return [p]

    def bc_y_min(arg, model):
        f, u, v, p, x, y = basic_symbols(arg, model)
        return [u, v]

    def bc_y_max(arg, model):
        f, u, v, p, x, y = basic_symbols(arg, model)
        return [u, v]

    def block(arg, model):
        f, u, v, p, x, y = basic_symbols(arg, model)
        return [u, v]

    domain1 = RectangleArea(low=[-5.0, -2.5], high=[-0.5, 2.5])
    domain2 = RectangleArea(low=[-0.5, -2.5], high=[0.5, -0.5])
    domain3 = RectangleArea(low=[-0.5, 0.5], high=[0.5, 2.5])
    domain4 = RectangleArea(low=[0.5, -2.5], high=[5.0, 2.5])

    x_min = RectangleArea(low=[-5.0, -2.5], high=[-5.0, 2.5])
    x_max = RectangleArea(low=[5.0, -2.5], high=[5.0, 2.5])
    y_min = RectangleArea(low=[-5.0, -2.5], high=[5.0, -2.5])
    y_max = RectangleArea(low=[-5.0, 2.5], high=[5.0, 2.5])

    x_min_block = RectangleArea(low=[-0.5, -0.5], high=[-0.5, 0.5])
    x_max_block = RectangleArea(low=[0.5, -0.5], high=[0.5, 0.5])
    y_min_block = RectangleArea(low=[-0.5, -0.5], high=[0.5, -0.5])
    y_max_block = RectangleArea(low=[-0.5, 0.5], high=[0.5, 0.5])

    pde = [
        Condition(inner, domain1), # 0 и 1 индекс соответствуют уравнению Навье-Стокса
        Condition(inner, domain2), # 3 и 4 индекс соответствуют уравнению Навье-Стокса
        Condition(inner, domain3), # 6 и 7 индекс соответствуют уравнению Навье-Стокса
        Condition(inner, domain4), # 9 и 10 индекс соответствуют уравнению Навье-Стокса

        Condition(bc_x_min, x_min),
        Condition(bc_x_max, x_max),
        Condition(bc_y_min, y_min),
        Condition(bc_y_max, y_max),

        Condition(block, x_min_block),
        Condition(block, x_max_block),
        Condition(block, y_min_block),
        Condition(block, y_max_block),
    ]

    return pde, input_dim, output_dim

conditions, input_dim, output_dim = navier_stocks_equation_with_block()

...

trainer = Trainer(
    ...,
    calc_loss=BigLambdaLossesRegularization(ignored_indexes=[0, 1, 3, 4, 6, 7, 9, 10]),
    ...
)
```

## LambdaLossesRegularization

```python
class callbacks.regularization_callback.regularization_callbacks.LambdaLossesRegularization(alpha: float)
```

Класс **LambdaLossesRegularization** добавляет при обучении регуляризацию выравнивающую градиенты. Основан на статье
[UNDERSTANDING AND MITIGATING GRADIENT FLOW PATHOLOGIES IN PHYSICS-INFORMED NEURAL NETWORKS](https://arxiv.org/abs/2001.04536).

> **Важно:**
> Данная регуляризация не адаптирована под решение систем дифференциальных уравнений и подходит только для одиночных
> уравнений.

### Параметры:

- **alpha** (float, по умолчанию 0.9) – указывает, насколько параметры регуляризации должны учитывать значение на
  предыдущем шаге оптимизации.

### Пример использования

```python

...

trainer = Trainer(
    ...,
    calc_loss=LambdaLossesRegularization(),
    ...
)
```

## NormalLossesRegularization

```python
class callbacks.regularization_callback.regularization_callbacks.NormalLossesRegularization()
```

Класс **NormalLossesRegularization** добавляет при обучении регуляризацию выравнивающую градиенты, в предположении, что
градиенты для каждой ошибки подчиняются нормальному распределению.

> **Важно:**
> Данная регуляризация не адаптирована под решение систем дифференциальных уравнений и подходит только для одиночных
> уравнений.

### Пример использования

```python

...

trainer = Trainer(
    ...,
    calc_loss=NormalLossesRegularization(),
    ...
)
```

## ConstantRegularization

```python
class callbacks.regularization_callback.regularization_callbacks.ConstantRegularization(lambdas: List[float])
```

Класс **ConstantRegularization** добавляет при обучении статическую регуляризацию.

### Параметры:

- **lambdas** (List[float]) – список параметров регуляризации соответствующих ошибок.

> **Важно:**
> Параметры регуляризации следует указывать в соответствии с порядком уравнений, которые были описаны в задаче.

### Пример использования

```python

...

trainer = Trainer(
    ...,
    calc_loss=NormalLossesRegularization(),
    ...
)
```