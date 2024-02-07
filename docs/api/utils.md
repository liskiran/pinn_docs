# Utils

```python
def configure_generators(conditions: List[Condition], generator_config: List[dict], default_gen: BaseGenerator = None):
```

Распределяет алгоритмы генерации по областям (conditions) в соответсвии c конфигурацией (genertor_config).

**Параметры **

- ** conditions ** (List[Condition]) - список областей, заданный в описании задачи
- ** generator_config ** (List[dict]) - список содержащий информацию о назначении алгоритмов генерации по областям.
- ** default_gen ** (BaseGenerator) - алгоритм генерации, используемый по умолчанию. Назначается на области, не упомянутые в generator config 

**Пример**

Задание списка областей в файле с проблемой

```python
inlet_reg = RectangleArea(low=[0.0, 0.0], high=[5.0, 1.0])
middle_reg = RectangleArea(low=[4.0, 1.0], high=[5.0, 2.0])
outlet_reg = RectangleArea(low=[4.0, 2.0], high=[9.0, 3.0])

inlet_bnd = RectangleArea(low=[0.0, 0.0], high=[0.0, 1.0])
outlet_bnd = RectangleArea(low=[9.0, 2.0], high=[9.0, 3.0]) 

wall1 = RectangleArea(low=[0.0, 0.0], high=[5.0, 0.0])
wall2 = RectangleArea(low=[0.0, 1.0], high=[4.0, 1.0])
wall3 = RectangleArea(low=[5.0, 0.0], high=[5.0, 2.0])

pde = [
        Condition(inner, inlet_reg),
        Condition(inner, middle_reg),
        Condition(inner, outlet_reg),

        Condition(inlet, inlet_bnd),
        Condition(outlet, outlet_bnd),

        Condition(wall, wall1),
        Condition(wall, wall2),
        Condition(wall, wall3)
    ]
```

В соответсвии с этим списком будет задаваться распределение алгоритмов генерации

```python
generator_domain = UniformGeneratorRect(n_points=5000,
                                            method='uniform')

generator_bound = UniformGeneratorRect(n_points=800,
                                           method='uniform')

generator_walls = UniformGeneratorRect(n_points=1500,
                                           method='uniform')                                      

generators_config = [
        {"condition_index": [0, 1, 2], "generator": generator_domain},
        {"condition_index": [3,4], "generator": generator_domain}
        ]

configure_generators(conditions, generators_config, default_gen=generator_walls)
```

Список [0,1,2] содержит индексы областей inlet_reg, middle_reg, outlet_reg (в порядке хранения в conditions), на которые необходимо назначить генератор generator_domain.
На область с индексами [3,4] нужно назначить генератор generator_bound.
Для оставшихся областей, не фигурирующих в generators_config, используется generator_walls, который указывается как аргумент default_gen.