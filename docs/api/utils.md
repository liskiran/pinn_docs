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

```python
generator_domain = UniformGeneratorRect(n_points=5000,
                                            method='uniform')

    generator_bound = UniformGeneratorRect(n_points=800,
                                           method='uniform')

    generators_config = [
        {"condition_index": [0, 1, 2], "generator": generator_domain},
    ]

    configure_generators(conditions, generators_config, default_gen=generator_bound)
```

Список [0,1,2] содержит индексы областей (в порядке хранения в conditions), на которые необходимо назначить нужный алгоритм генерации.