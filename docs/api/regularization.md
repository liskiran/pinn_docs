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

> **Важно**
> В данный список следует добавлять индексы уравнений, которые являются основными для вашей задачи (например, при
> решении задачи движения вязкой несжимаемой жидкости, в данный список следует добавить индексы уравнений Навье-Стокса,
> а уравнение неразрывности, граничные и начальные условия не добавлять)

### Пример использования

```python
conditions, input_dim, output_dim = src.problems.navier_stocks_equation_with_block()

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

> **Важно**
> Данная регуляризация не адаптирована под решение систем дифференциальных уравнений и подходит только для одиночных
> уравнений.

### Параметры:

- **alpha** (float, по умолчанию 0.9) – указывает, насколько параметры регуляризации должны учитывать значение на
  предыдущем шаге оптимизации.

### Пример использования

```python
conditions, input_dim, output_dim = src.problems.problem_2D1C_heat_equation()

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

> **Важно**
> Данная регуляризация не адаптирована под решение систем дифференциальных уравнений и подходит только для одиночных
> уравнений.

### Пример использования

```python
conditions, input_dim, output_dim = src.problems.problem_2D1C_heat_equation()

...

trainer = Trainer(
    ...,
    calc_loss=NormalLossesRegularization(),
    ...
)
```

## NormalLossesRegularization

```python
class callbacks.regularization_callback.regularization_callbacks.ConstantRegularization(lambdas: List[float]

)
```

Класс **ConstantRegularization** добавляет при обучении статическую регуляризацию.

### Параметры:

- **lambdas** (List[float]) – список параметров регуляризации соответствующих ошибок.

> **Важно**
> Параметры регуляризации следует указывать в соответствии с порядком уравнений, которые были описаны в задаче.

### Пример использования

```python
conditions, input_dim, output_dim = src.problems.problem_2D1C_heat_equation()

...

trainer = Trainer(
    ...,
    calc_loss=NormalLossesRegularization(),
    ...
)
```