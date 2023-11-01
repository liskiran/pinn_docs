## PretrainedEnsemble
    CLASS neural_network.ensemble.PretrainedEnsemble(layers_all: List, pretrained_models: List[torch.nn.Module])

Класс, позволяющий создать ансамбль моделей на основе списка **предобученных моделей** подходом [stacking](https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/). После иницализации работает как torch.nn.Module. 

**Параметры**
- **layers_all** (List) – список слоев FNN модели, которая будет аггрегировать выходы предобученных моделей.
- **pretrained_models** (List[torch.nn.Module]) – список предобученных моделей. 
    

**Методы**

    forward(input: torch.Tensor)

Выполняет forward-pass ансамбля. 

## TrainableEnsemble
    CLASS neural_network.ensemble.TrainableEnsemble(train_model_list: List[torch.nn.Module]) 

Класс, позволяющий создать ансамбль из еще не обученых моделей, для обучения ансамбля с нуля. После иницализации работает как torch.nn.Module. 
 
Если вы используете этот класс, то для обучения моделей из ансамбля будут применяется один и тот же optimizer, scheduler и generator. Для обучения более гибкого ансамбля с нуля используется [EnsembleTrainer](#ensembletrainer) и пример #TODO

**Параметры**

- **train_model_list**(List[torch.nn.Module]) – список моделей для ансамблирования.

**Методы**

    forward(input: torch.Tensor)

Выполняет forward-pass ансамбля.


## EnsembleInstance
    CLASS neural_network.ensemble.EnsembleInstance(model_name: str, pinn: PINN, optimizer: torch.optim.Optimizer, scheduler: torch.optim.LRScheduler)

Класс, хранящий нужную информацию одной модели для обучения гибкого ансамбля. Создается в [ensemble_builder](#ensemble_builder). 

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

Модификация класса [Trainer](trainer.md) для обучения гибкого ансамбля. 

**Параметры**
- **ensemble_config**(List[[EnsembleInstance](#ensembleinstance)]) - информация для ансабля полученная с помощью [ensemble_builder](#ensemble_builder). 
- **output_dim**(int) - размер выхода ансамбля. 
- **\*\*kwargs** – аргументы [Trainer](trainer.md).

**Методы**
    
    train -> None
    
Обучает ансамбль. 

## ensemble_builder
    FUNCTION neural_network.ensemble.ensemble_builder
Вспомогательная функция для сборки [EnsembleInstance](#ensembleinstance)

**Параметры**
- **models**(List[torch.nn.Module]) - список моделей для ансамблирования.
- **generatorss_domain**(List[BaseGenerator]) - список генераторов (*в области*) для каждой модели.
- **generators_bound**(List[BaseGenerator]) - список генераторов (*на границе*) для каждой модели.
- **condition_idx**(List[int]) - список индексов условий.
- **optimizers**(List[Optimizer]) - список оптимизаторов
- **schedulers**(List[LRScheduler]) - список scheduler'ов.
- **conditions**(List[Condition]) - список условий. 

