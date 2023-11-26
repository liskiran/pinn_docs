```python
class Trainer
```

### Методы

Инициализация класса Trainer

```python
def __init__(self,
    pinn: PINN,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    num_epochs: int = 1000,
    update_grid_every: int = 1,
    calc_loss: Callable[[], List[torch.Tensor]] = None,
    callbacks_organizer: CallbacksOrganizer = None,
    mixed_training: bool = False
) -> None:
```

### Параметры:

- `pinn: PINN` - Объект класса `PINN` содержащий в себе информацию об модели, уравнениях и функции ошибки.
- `optimizer: torch.optim.Optimizer` - Оптимизатор необходимый для обучения нейронной сети
- `scheduler: torch.optim.lr_scheduler.LRScheduler` - Шедулер для рбновления learning rate нейронной сети
- `num_epochs: int` (default: 1000) - Количество эпох для обучения нейронной сети
- `update_grid_every: int` (default: 1) - Обновление сетки для изменения пространственно-временных координат.
- `calc_loss: Callable[[], List[torch.Tensor]]` (default: None) - Функция ошибки
- `callbacks_organizer: CallbacksOrganizer` (default: None) - Объект класса `CallbacksOrganizer` для управления визуализацией и информацией во время обучения
- `mixed_training: bool` (default: False)

---

```python
def train(self):
```

Запуск обучения PINN

---

```python
def _train_epoch(self):
```

Одна итерация обучения PINN

---

```python
def default_calc_loss(self):
```

Подсчет функции потерь: Состоит из потерь на невязках уравнения, граничных и начальных условий