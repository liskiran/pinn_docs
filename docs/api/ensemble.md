## PretrainedEnsemble

    CLASS neural_network.ensemble.PretrainedEnsemble(layers_all: List, pretrained_models: List[torch.nn.Module])

Класс, позволяющий создать ансамбль моделей на основе списка **предобученных моделей** подходом [stacking](https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/). После иницализации работает как torch.nn.Module.

**Параметры**

- **layers_all** (List) – список слоев FNN модели, которая будет аггрегировать выходы предобученных моделей.
- **pretrained_models** (List[torch.nn.Module]) – список предобученных моделей.

**Методы**

    forward(input: torch.Tensor)

Выполняет прямой проход (forward-pass) ансамбля.

**Примеры использования**

```python
from neural_network.ensemble import PretrainedEnsemble

# Определение списка слоев и предобученных моделей
layers_all = [1, 32, 1]
pretrained_models = [model1, model2, model3]

# Создание ансамбля из предобученных моделей
ensemble = PretrainedEnsemble(layers_all, pretrained_models)

# Выполнение прямого ансамбля
output = ensemble.forward(input_tensor)

```

## TrainableEnsemble

    CLASS neural_network.ensemble.TrainableEnsemble(train_model_list: List[torch.nn.Module])

Класс, позволяющий создать ансамбль из еще не обученых моделей, для обучения ансамбля с нуля. После иницализации работает как torch.nn.Module.

Если вы используете этот класс, то для обучения моделей из ансамбля будут применяется один и тот же оптимизатор и генератор. Для обучения более гибкого ансамбля с нуля используется [EnsembleTrainer](#ensembletrainer)

**Параметры**

- **train_model_list**(List[torch.nn.Module]) – список моделей для ансамблирования.

**Методы**

    forward(input: torch.Tensor)

Выполняет прямой проход (forward-pass) ансамбля.
**Примеры использования:**

```python
from neural_network.ensemble import TrainableEnsemble

# Определение списка моделей для ансамблирования
models = [model1, model2, model3]

# Создание ансамбля из еще не обученных моделей
ensemble = TrainableEnsemble(models)

# Выполнение прямого прохода ансамбля
output = ensemble(input_tensor)

```

## EnsembleInstance

    CLASS neural_network.ensemble.EnsembleInstance(model_name: str, pinn: PINN, optimizer: torch.optim.Optimizer, scheduler: torch.optim.LRScheduler)

Класс EnsembleInstance используется для хранения информации об одной модели для обучения ансамбля, в котором для каждой модели задаются произвольные оптимизатор (optimizer), генераторы (generator), курс обучения (scheduler). Он создается в функции [ensemble_builder](#ensemble_builder). При инициализации этот класс принимает четыре аргумента:

**Параметры**

- **model_name**(str) – название текущей модели. Используется для сохранения и логгирования.
- **pinn**(PINN) – экземпляр PINN для текущей модели. Должен содержать информацию о генераторах и геометрии.
- **optimizer**(torch.optim.Optimizer) - оптимизатор для модели, например torch.optim.Adam.
- **scheduler**(torch.optim.lr_scheduler.LRScheduler) – scheduler для модели, например.

**Методы**

    get_parameters -> Tuple[str, PINN, Optimizer, LRScheduler]

Возвращает сохраненные параметры для модели.

## EnsembleTrainer

    CLASS neural_network.ensemble.EnsembleTrainer(ensemble_config: List[EnsembleInstance], output_dim: int, **kwargs)

Модификация класса [Trainer](trainer.md) для обучения ансамбля, в котором для каждой модели задаются произвольные оптимизатор (optimizer), генераторы (generator), курс обучения (scheduler).

**Параметры**

- **ensemble_config**(List[[EnsembleInstance](#ensembleinstance)]) - информация для ансабля полученная с помощью [ensemble_builder](#ensemble_builder).
- **output_dim**(int) - размер выхода ансамбля.
- **\*\*kwargs** – аргументы [Trainer](trainer.md).

**Методы**

    train -> None

Обучает ансамбль.

**Примеры использования**:

Полноценный пример в разделе [Ансамблирование моделей](/docs/guide/ensemble.ipynb).

```python
callbacks_orgaziner = EnsembleCallbacksOrganizer(callbacks)

trainer = EnsembleTrainer(
    ensemble_config,
    callbacks_organizer=callbacks_orgaziner,
    num_epochs=1000,
    output_dim=1,
)

trainer.train()
```

## ensemble_builder

    FUNCTION neural_network.ensemble.ensemble_builder

Вспомогательная функция для подготовки нескольких [EnsembleInstance](#ensembleinstance) из списка моделей, генераторов, оптимизаторов.

**Параметры**

- **models**(List[torch.nn.Module]) - список моделей для ансамблирования.
- **generatorss_domain**(List[BaseGenerator]) - список генераторов (_в области_) для каждой модели.
- **generators_bound**(List[BaseGenerator]) - список генераторов (_на границе_) для каждой модели.
- **condition_idx**(List[int]) - список индексов условий.
- **optimizers**(List[Optimizer]) - список оптимизаторов
- **schedulers**(List[LRScheduler]) - список scheduler'ов.
- **conditions**(List[Condition]) - список условий.

**Примеры использования:**

Полноценный пример в разделе [Ансамблирование моделей](/docs/guide/ensemble.ipynb).

```python
models = [
    ResNet([input_dim, 32, 64, 64, 32, output_dim]),
    FNN([input_dim, 128, 256, 128, output_dim]),
    XavierFNN([input_dim, 128, 128, 128, 128, output_dim]),
]

generators_domain = [
    UniformGeneratorRect(n_points=5000, method="uniform") for _ in range(3)
]

generators_boundary = [
    UniformGeneratorRect(n_points=500, method="uniform") for _ in range(3)
]

optimizers = [torch.optim.Adam(model.parameters()) for model in models]

schedulers = [ExponentialLR(optimizer=opt, gamma=0.999) for opt in optimizers]

ensemble_config = ensemble_builder(
    models,
    generators_domain,
    generators_boundary,
    [0, 1, 2, 3],
    optimizers,
    schedulers,
    conditions,
)

#Используем полученную конфигурацию для обучения:

trainer = EnsembleTrainer(
    ensemble_config,
    callbacks_organizer=callbacks_orgaziner,
    num_epochs=1000,
    output_dim=output_dim,
)

trainer.train()
```
