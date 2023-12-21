# PINN

[PINN](#pinn-1) (Physics-Informed Neural Network)–  это класс, предназначенный для удобной работы с физически-информированными нейронными сетями. Он хранит необходимые объекты для обучения, такие как модель, уравнение и граничные/начальные условия, функцию потерь. 

Данный класс реализует методы необходимые для обучения нейросети: подсчет функции потерь, обновление значений точек обучающей выборки.

## PINN
    CLASS src.PINN.PINN(model: torch.nn.Module, conditions: List[Condition], loss_function: torch.nn.modules.loss)

**Параметры**

- **model** (torch.nn.Module) – PyTorch модель. Например, [FNN](neural_network.md#feed-forward-neural-network). 
- **conditions** (List[[Condition](condition.md)]) – список условий, каждое из которых характеризует отдельное уравнение или граничное/начальное условие. 
- **loss_function** (torch.nn.modules.loss) – функция потерь. По умолчанию [torch.nn.MSELoss](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html).
    

**Методы**

    update_data() -> None

Для каждого условия обновляет точки сетки, используя метод [update_points].

    calculate_condition_loss(cond: Condition) -> List[torch.Tensor]

Считает уже выбранную функцию потерь на конкретном условии, которое передается в аргумент.

    calculate_loss() -> List[torch.Tensor]

Считает функцию потерь на всех заданнных в PINN условиях. Возвращает итоговые суммарные потери. 

    calculate_loss_on_points(cond: Condition, points: torch.Tensor) -> List[torch.Tensor]

Считает функцию потерь для конкретного условия при заданном наборе точек. Используется, например, для отрисовки потерь на другой, отличной от сетки для обучения сетке. 

**Примеры использования**

```python
# Инициализируем проблему
conditions, input_dim, output_dim = src.problems.navier_stocks_equation_with_block() 

# Фиксируем ускоритель (GPU/CPU)
set_device()

# Создаем модель
model = FNN(layers_all=[input_dim, 128, 128, 128, 128, output_dim])

# Создаем необходимые генераторы
generator_domain = UniformGeneratorRect(n_points=500,
                                        n_dims=input_dim,
                                        method='uniform',
                                        add_points=50)

generator_bound = UniformGeneratorRect(n_points=50,
                                        n_dims=input_dim,
                                        method='uniform',
                                        add_points=50)

generators_config = [
    {"condition_index": [0, 1, 2, 3], "generator": generator_domain},
]

configure_generators(conditions, generators_config, default_gen=generator_bound)

# Создаем объект PINN
pinn = PINN(model=model, conditions=conditions)

optimizer = torch.optim.Adam(model.parameters())

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)

trainer = Trainer(
        pinn=pinn,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=100,
        update_grid_every=10,
    )

trainer.train()
```
