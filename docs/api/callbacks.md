
## Grid
    CLASS callbacks.heatmap.Grid(low: np.array, high: np.array, n_points: Union[Sequence[int], int])

Класс, позволяющий создать сетку точек в указанной области. Сетку можно использовать для построения тепловых карт и расчета значения функций на структурированном наборе точек

**Параметры**

- **low** (np.array) : массив со значениями нижних границ для каждой оси,
- **high** (np.array) : np.array со значениями верхних границ для каждой оси,
- **n_points** (Union[Sequence[int], int]) : массив со значениями количества точек по каждой оси / общее количество точек на весь Grid.

**Методы**

- **from_pinn(cls, pinn: PINN, n_points: Union[Sequence[int], int])** : Создает объект Grid на основе PINN и указанного количества точек pinn
- **from_condition(cls, condition, n_points: Union[Sequence[int], int])** : Создает объект Grid на основе условий задачи и указанного количества точек.
- **smart_volume(low, high, total_points: int) -> Sequence[int]** : Статический метод для расчета количества точек по каждой оси на основе указанного числа точек.

**Примеры использования**

Построение сетки с указанными верхней и нижней границами и количеством точек по осям:
```python
from src.callbacks.heatmap import Grid
grid = Grid(low=[0, 0], high=[1, 1], n_points=[100, 100])
```
Построение сетки с указанными верхней и нижней границами и общим количеством точек:
```python
from src.callbacks.heatmap import Grid
grid = Grid(low=[0, 0], high=[1, 1], n_points=10001)
```
Построение сетки на основе экземпляра класса PINN и указанного количества точек:
```python
from src.callbacks.heatmap import Grid
from src.problems import problem_2D1C_heat_equation

conditions, input_dim, output_dim = problem_2D1C_heat_equation()
model = FNN(layers_all=[input_dim, 128, 128, 128, output_dim])
pinn = PINN(model=model, conditions=conditions)
grid = Grid.from_pinn(pinn, 80001)
```
Построение сетки на основе условий задачи и указанного количества точек:
```python
from src.callbacks.heatmap import Grid
from src.problems import problem_2D1C_heat_equation

conditions, input_dim, output_dim = problem_2D1C_heat_equation()
grid = Grid.from_condition(conditions, 10001)
```


  

## BaseImageSave
    CLASS callbacks.heatmap.BaseImageSave(self, save_dir: str, period: int, save_mode: str = 'html')
    
Абстрактный класс для наследования класса BaseHeatmap

**Параметры**

- **save_dir** (str) : директория для сохранения графика,
- **period** (int) : период сохранения графиков,
- **save_mode** (str) : режим сохранения. Режимы сохранения:
    - “html” : сохраняет каждую тепловую карту в указанной директории в формате html;
    - “png” : сохраняет тепловые карты в указанной директории в формате png;
    - “pt” : сохраняет точки, по которым можно построить данную тепловую карту;
    - “show” : открывает каждую тепловую карту в браузере в интерактивном режиме, далее ее можно сохранить вручную.

**Методы**

- **save_fig(self, fig, file_name: str)** : Сохраняет или отрисовывает график в одном из режимов.
- **save_pt(self, fig, file: str)** : Метод для сохранения точек графика.
  
## BaseHeatmap
    CLASS callbacks.heatmap.BaseHeatmap(self, grid: Grid, save_dir: str, period: int = 500, save_mode: Literal["html", "png", "pt", "show"] = "html", output_index: int = 0, min = None, max = None, x_name: float  = "x", y_name: str  = "y", z_name: str = "z")
    
Абстрактный класс для наследования другими классами тепловых карт.

**Параметры**

- **save_dir** (str) : директория для сохранения графика,
- **grid** (str) : объект класса Grid,
- **period** (int) : период сохранения графиков,
- **save_mode** (str) : режим сохранения. Режимы сохранения:
    - “html” : сохраняет каждую тепловую карту в указанной директории в формате html;
    - “png” : сохраняет тепловые карты в указанной директории в формате png;
    - “pt” : сохраняет точки, по которым можно построить данную тепловую карту;
    - “show” : открывает каждую тепловую карту в браузере в интерактивном режиме, далее ее можно сохранить вручную,
- **output_index** (int) : номер уравнения, график которого наобходимо построить,
- **min** (float | None) : минимальное значение colorbar'а,
- **max** (float | None) : максимальное значение colorbar'а,
- **x_name** (str) : название оси абсцисс ("x" по умолчанию),
- **y_name** (str) : название оси ординат ("y" по умолчанию),
- **z_name** (str) : название оси аппликат ("z" по умолчанию).

**Методы**

- **draw(self, values: torch.Tensor, plot_name: str, file_name: str = None, min = None, max = None, x_name = "z", y_name = "y", z_name="z")** : В зависимости от размерности вызывают одну из функций для построения тепловой карты.
- **dict_data(self, fig)** : Метод для сохранения точек графика.
- **draw_3D(self, values: torch.Tensor, plot_name: str, file_name: str = None, min = None, max = None, x_name="x", y_name="y", z_name="z")** : Строит трехмерную тепловую карту и сохраняет ее в выбранном формате.
- **draw_2D(self, values: torch.Tensor, plot_name: str, file_name: str = None, min = None, max = None, x_name = "x", y_name = "y")** : Строит двумерную тепловую карту и сохраняет ее в выбранном формате.
- **draw_1D(self, values: torch.Tensor, plot_name: str, file_name: str = None, min = None, max = None, x_name = "x")** : Строит одномерную тепловую карту и сохраняет ее в выбранном формате.
## HeatmapError

    CLASS callbacks.heatmap.HeatmapError(self, grid: Grid, save_dir: str, period: int = 500, save_mode: Literal["html", "png", "pt", "show"] = "html", output_index: int = 0, min = None, max = None, x_name: float  = "x", y_name: str  = "y", z_name: str = "z")
    
Функция обратного вызова (callback) для создания тепловых карт ошибок во время обучения модели.

**Параметры**

- **save_dir** (str) : директория для сохранения графика,
- **grid** (str) : объект класса Grid,
- **solution** : функция точного решения,
- **period** (int) : период сохранения графиков,
- **save_mode** (str) : режим сохранения. Режимы сохранения:
    - “html” : сохраняет каждую тепловую карту в указанной директории в формате html;
    - “png” : сохраняет тепловые карты в указанной директории в формате png;
    - “pt” : сохраняет точки, по которым можно построить данную тепловую карту;
    - “show” : открывает каждую тепловую карту в браузере в интерактивном режиме, далее ее можно сохранить вручную,
- **output_index** (int) : номер уравнения, график которого наобходимо построить,
- **min** (float | None) : минимальное значение colorbar'а,
- **max** (float | None) : максимальное значение colorbar'а,
- **x_name** (str) : название оси абсцисс ("x" по умолчанию),
- **y_name** (str) : название оси ординат ("y" по умолчанию),
- **z_name** (str) : название оси аппликат ("z" по умолчанию).

**Методы**

- **__call__(self, trainer: Trainer)** : Использование функции обратного вызова (callback) при обучении модели.

**Пример использования**
```python
from src.callbacks.heatmap import HeatmapError

def exact_solution(args, a=1, b=0.5, alpha=0.5, beta=10, gamma=0.7):
    return torch.exp(-gamma ** 2 * args[:, 1]) * (a * torch.cos(alpha * args[:, 0]) + b * torch.sin(beta * args[:, 0]))
grid = Grid.from_pinn(pinn, 80001)
save_dir = "reports"
callbacks = [HeatmapError(save_dir, grid=grid, solution=exact_solution, period=500, save_mode='html')]
trainer = Trainer(
    pinn=pinn,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=5000,
    update_grid_every=100,
    callbacks=callbacks,
)
trainer.train()
```

## HeatmapPrediction

    CLASS callbacks.heatmap.HeatmapPrediction(self, grid: Grid, save_dir: str, period: int = 500, save_mode: Literal["html", "png", "pt", "show"] = "html", output_index: int = 0, min = None, max = None, x_name: float  = "x", y_name: str  = "y", z_name: str = "z")
    
Функция обратного вызова (callback) для создания тепловых карт решения, полученного моделью.

**Параметры**

- **save_dir** (str) : директория для сохранения графика,
- **grid** (str) : объект класса Grid,
- **period** (int) : период сохранения графиков,
- **save_mode** (str) : режим сохранения. Режимы сохранения:
    - “html” : сохраняет каждую тепловую карту в указанной директории в формате html;
    - “png” : сохраняет тепловые карты в указанной директории в формате png;
    - “pt” : сохраняет точки, по которым можно построить данную тепловую карту;
    - “show” : открывает каждую тепловую карту в браузере в интерактивном режиме, далее ее можно сохранить вручную,
- **output_index** (int) : номер уравнения, график которого наобходимо построить,
- **min** (float | None) : минимальное значение colorbar'а,
- **max** (float | None) : максимальное значение colorbar'а,
- **x_name** (str) : название оси абсцисс ("x" по умолчанию),
- **y_name** (str) : название оси ординат ("y" по умолчанию),
- **z_name** (str) : название оси аппликат ("z" по умолчанию).

**Методы**

- **__call__(self, trainer: Trainer)** : Использование функции обратного вызова (callback) при обучении модели.

**Пример использования**
```python
from src.callbacks.heatmap import HeatmapPrediction

grid = Grid.from_pinn(pinn, 80001)
save_dir = "reports"
callbacks = [HeatmapPrediction(save_dir, grid=grid, period=500, save_mode='png')]
trainer = Trainer(
    pinn=pinn,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=5000,
    update_grid_every=100,
    callbacks=callbacks,
)
trainer.train()
```
## PlotHeatmapSolution

    CLASS callbacks.heatmap.PlotHeatmapSolution(self, save_dir: str, grid: Grid, solution: Callable[[torch.Tensor], torch.Tensor],
                 save_mode: str = 'html', min = None, max = None, x_name: float  = "x", y_name: str  = "y", z_name: str = "z"))
    
Класс для построения графика точного решения.
> **_Важно:_** Это не функция обратного вызова (callback)!

**Параметры**

- **save_dir** (str) : директория для сохранения графика,
- **grid** (str) : объект класса Grid,
- **solution** : функция точного решения,
- **save_mode** (str) : режим сохранения. Режимы сохранения:
    - “html” : сохраняет каждую тепловую карту в указанной директории в формате html;
    - “png” : сохраняет тепловые карты в указанной директории в формате png;
    - “pt” : сохраняет точки, по которым можно построить данную тепловую карту;
    - “show” : открывает каждую тепловую карту в браузере в интерактивном режиме, далее ее можно сохранить вручную,
- **min** (float | None) : минимальное значение colorbar'а,
- **max** (float | None) : максимальное значение colorbar'а,
- **x_name** (str) : название оси абсцисс ("x" по умолчанию),
- **y_name** (str) : название оси ординат ("y" по умолчанию),
- **z_name** (str) : название оси аппликат ("z" по умолчанию).

**Пример использования**
```python
from src.callbacks.heatmap import PlotHeatmapSolution

def exact_solution(args, a=1, b=0.5, alpha=0.5, beta=10, gamma=0.7):
    return torch.exp(-gamma ** 2 * args[:, 1]) * (a * torch.cos(alpha * args[:, 0]) + b * torch.sin(beta * args[:, 0]))
grid = Grid.from_pinn(pinn, 80001)
save_dir = "reports"
PlotHeatmapSolution(save_dir, grid=grid, solution=exact_solution, save_mode='show')
```

## MeshHeatmapPrediction

    CLASS callbacks.heatmap.MeshHeatmapPrediction(self, save_dir: str, period: int, points: torch.Tensor, save_mode: str = 'html', output_index: int = 0, min = None, max = None, x_name: float  = "x", y_name: str  = "y", z_name: str = "z"))
Функция обратного вызова (callback) для построения тепловой карты (heatmap) решения на mesh сетке.

**Параметры**

- **save_dir** (str) : директория для сохранения графика,
- **period** (int) : период сохранения графиков,
- **points** (torch.Tensor) : точки для построения графика,
- **save_mode** (str) : режим сохранения. Режимы сохранения:
    - “html” : сохраняет каждую тепловую карту в указанной директории в формате html;
    - “png” : сохраняет тепловые карты в указанной директории в формате png;
    - “pt” : сохраняет точки, по которым можно построить данную тепловую карту;
    - “show” : открывает каждую тепловую карту в браузере в интерактивном режиме, далее ее можно сохранить вручную,
- **output_index** (int) : номер уравнения, график которого наобходимо построить,
- **min** (float | None) : минимальное значение colorbar'а,
- **max** (float | None) : максимальное значение colorbar'а,
- **x_name** (str) : название оси абсцисс ("x" по умолчанию),
- **y_name** (str) : название оси ординат ("y" по умолчанию),
- **z_name** (str) : название оси аппликат ("z" по умолчанию).

**Методы**

- **__call__(self, trainer: Trainer)** : Использование функции обратного вызова (callback) при обучении модели.

**Пример использования**
```python
from src.neural_network import FNN
from src.PINN import PINN
from src.callbacks.heatmap import MeshHeatmapPrediction

conditions, input_dim, output_dim = src.problems.real_navier_stocks(Re=60, mesh_file_path ='../nsu_1.pt', mass=True)
model = FNN(layers_all=[input_dim, 64, output_dim])
pinn = PINN(model=model, conditions=conditions)
save_dir = "reports"
callbacks = [MeshHeatmapPrediction(save_dir, 1000,  pinn.conditions[0].geometry.points[::10], min=-5, max=5, output_index=0),
            MeshHeatmapPrediction(save_dir, 1000,  pinn.conditions[0].geometry.points[::10], min=-5, max=5, output_index=1),
            MeshHeatmapPrediction(save_dir, 1000,  pinn.conditions[0].geometry.points[::10], min=-5, max=5, output_index=2),
            MeshHeatmapPrediction(save_dir, 1000,  pinn.conditions[0].geometry.points[::10], min=-5, max=5, output_index=3)]
trainer = Trainer(
    pinn=pinn,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=5000,
    update_grid_every=100,
    callbacks=callbacks)
trainer.train()
```

## BasicCurve
    CLASS callbacks.curve.BasicCurve(self, save_dir: str, period: int = 500, save_mode: str = 'html', log_scale: bool = True, x_title: str = "Epoch", y_title: str = "Loss")
Абстрактный класс для наследования другими классами кривых.

**Параметры**

- **save_dir** (str) : директория для сохранения графика,
- **period** (int) : период сохранения графиков,
- **save_mode** (str) : режим сохранения. Режимы сохранения:
    - “html” : сохраняет каждую тепловую карту в указанной директории в формате html;
    - “png” : сохраняет тепловые карты в указанной директории в формате png;
    - “pt” : сохраняет точки, по которым можно построить данную тепловую карту;
    - “show” : открывает каждую тепловую карту в браузере в интерактивном режиме, далее ее можно сохранить вручную,
- **log_scale** (bool) : флаг, определяющий будет ли ось OY логарифмирована,
- **x_title** (str) : название оси абсцисс (по умочанию "Epoch"),
- **y_title** (str) : название оси ординат (по умолчанию "Loss"),
- **metric_history** (list) : список для сохранения истории значений метркии.

**Методы**

- **draw(self, values: Sequence, v_names: [str], coord: np.ndarray, plot_title: str)** : Строит и сохраняет кривую.
- **dict_data(fig)** : Сохраняет данные графика в виде словаря.
- **init_curve_names(self, conditions)** : Нзначает названия кривым в зависимости от conditions,
- **reset(self, new_save_dir: str = None)** : Меняет название директории для сохранения графика.

## GridResidualCurve
    CLASS callbacks.curve.GridResidualCurve(self, save_dir: str, grid: Grid, period=100, save_mode='html', log_scale: bool = True, condition_index=0)
Функция обратного вызова (callback) для построения кривых обучения на сетке Grid.

> **_Важно:_** Убедитесь, что Grid соответствует conditions.

**Параметры**

- **save_dir** (str) : директория для сохранения графика,
- **grid**(Grid) : Объект класса Grid,
- **period** (int) : период сохранения графиков,
- **save_mode** (str) : режим сохранения. Режимы сохранения:
    - “html” : сохраняет каждую тепловую карту в указанной директории в формате html;
    - “png” : сохраняет тепловые карты в указанной директории в формате png;
    - “pt” : сохраняет точки, по которым можно построить данную тепловую карту;
    - “show” : открывает каждую тепловую карту в браузере в интерактивном режиме, далее ее можно сохранить вручную,
- **log_scale** (bool) : флаг, определяющий будет ли ось OY логарифмирована,
- **condition_index** (int) : номер условия для построения графика.

**Методы**

- **__call__(self, trainer: Trainer) -> None** : Использование функции обратного вызова (callback) при обучении модели.

**Пример использования**
```python
from src.callbacks.curve import GridResidualCurve

grid = Grid.from_pinn(pinn, 80001)
save_dir = "reports"
callbacks = [GridResidualCurve(save_dir, grid=grid, period=100, save_mode='html')]
trainer = Trainer(
    pinn=pinn,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=5000,
    update_grid_every=100,
    callbacks=callbacks,
)
trainer.train()
```

## TrainingCurve
    CLASS callbacks.curve.TrainingCurve(self, save_dir: str, period=100, save_mode='html', log_scale: bool = True)
Функция обратного вызова (callback) для построения кривых обучения по лоссам модели.

**Параметры**

- **save_dir** (str) : директория для сохранения графика,
- **period** (int) : период сохранения графиков,
- **save_mode** (str) : режим сохранения. Режимы сохранения:
    - “html” : сохраняет каждую тепловую карту в указанной директории в формате html;
    - “png” : сохраняет тепловые карты в указанной директории в формате png;
    - “pt” : сохраняет точки, по которым можно построить данную тепловую карту;
    - “show” : открывает каждую тепловую карту в браузере в интерактивном режиме, далее ее можно сохранить вручную,
- **log_scale** (bool) : флаг, определяющий будет ли ось OY логарифмирована.

**Методы**

- **__call__(self, trainer: Trainer) -> None** : Использование функции обратного вызова (callback) при обучении модели.

**Пример использования**
```python
from src.callbacks.curve import TrainingCurve

save_dir = "reports"
callbacks = [TrainingCurve(save_dir, period=100, save_mode='png')]
trainer = Trainer(
    pinn=pinn,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=5000,
    update_grid_every=100,
    callbacks=callbacks,
)
trainer.train()
```

## LearningRateCurve
    CLASS callbacks.curve.LearningRateCurve(self, save_dir: str, period=100, save_mode='html', log_scale: bool = True)
Функция обратного вызова (callback) для построения кривой learning rate.

**Параметры**

- **save_dir** (str) : директория для сохранения графика,
- **period** (int) : период сохранения графиков,
- **save_mode** (str) : режим сохранения. Режимы сохранения:
    - “html” : сохраняет каждую тепловую карту в указанной директории в формате html;
    - “png” : сохраняет тепловые карты в указанной директории в формате png;
    - “pt” : сохраняет точки, по которым можно построить данную тепловую карту;
    - “show” : открывает каждую тепловую карту в браузере в интерактивном режиме, далее ее можно сохранить вручную,
- **log_scale** (bool) : флаг, определяющий будет ли ось OY логарифмирована.

**Методы**

- **__call__(self, trainer: Trainer) -> None** : Использование функции обратного вызова (callback) при обучении модели.

**Пример использования**

```python
from src.callbacks.curve import LearningRateCurve

save_dir = "reports"
callbacks = [LearningRateCurve(save_dir, period=500, log_scale=False, save_mode='html')]
trainer = Trainer(
    pinn=pinn,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=5000,
    update_grid_every=100,
    callbacks=callbacks,
)
trainer.train()
```

## ErrorCurve
    CLASS callbacks.curve.ErrorCurve(self, save_dir: str, solution: Callable[[torch.Tensor], torch.Tensor], period=100, save_mode='html',
                 log_scale: bool = True)
Функция обратного вызова (callback) для построения кривой ошибки.

**Параметры**

- **save_dir** (str) : директория для сохранения графика,
- **solution** (Callable) : функция точного решения,
- **period** (int) – период сохранения графиков,
- **save_mode** (str) : режим сохранения. Режимы сохранения:
    - “html” : сохраняет каждую тепловую карту в указанной директории в формате html;
    - “png” : сохраняет тепловые карты в указанной директории в формате png;
    - “pt” : сохраняет точки, по которым можно построить данную тепловую карту;
    - “show” : открывает каждую тепловую карту в браузере в интерактивном режиме, далее ее можно сохранить вручную,
- **log_scale** (bool) : флаг, определяющий будет ли ось OY логарифмирована.

**Методы**

- **__call__(self, trainer: Trainer) -> None** : Использование функции обратного вызова (callback) при обучении модели.

**Пример использования**

```python
from src.callbacks.curve import ErrorCurve

save_dir = "reports"
callbacks = [ErrorCurve(save_dir, solution=exact_solution, period=100)]
trainer = Trainer(
    pinn=pinn,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=5000,
    update_grid_every=100,
    callbacks=callbacks,
)
trainer.train()
```

## ProgressBar
    CLASS callbacks.progress.ProgressBar(self, template: str, period: int = 10)
Функция обратного вызова (callback) для обновления Progress Bar.

**Параметры**

- **template** (str) : шаблон строки, выводимой в консоль,
- **period** (int) : периодичность использования функции обратного вызова (callback).

**Методы**

- **__call__(self, trainer: Trainer) -> None** : Использование функции обратного вызова (callback) при обучении модели.
- **make_message(self, trainer: Trainer) -> str** : Создание строки, выводимой в консоль.
## TqdmBar
    CLASS callbacks.progress.TqdmBar(self, template: str, period: int = 10)
Функция обратного вызова (callback) для обновления Tqdm progress bar.

**Параметры**

- **template** (str) : шаблон строки, выводимой в консоль,
- **period** (int) : периодичность использования функции обратного вызова (callback).

**Методы**

- **__call__(self, trainer: Trainer) -> None** : Использование функции обратного вызова (callback) при обучении модели.

**Пример использования**
```python
from src.callbacks.progress import TqdmBar

callbacks = [TqdmBar('Epoch {epoch} lr={lr:.2e} Loss={loss_eq} Total={total_loss:.2e}')]
trainer = Trainer(
    pinn=pinn,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=5000,
    update_grid_every=100,
    callbacks=callbacks,
)
trainer.train()
```
## SaveModel
    CLASS callbacks.save.SaveModel(self, save_path: str, period: int = 1000)
Функция обратного вызова (callback) для сохранения модели.

**Параметры**

- **save_path** (str) : путь для сохранения файла (включая название),
- **period** (int) : периодичность использования функции обратного вызова (callback).

**Методы**

- **__call__(self, trainer: Trainer) -> None** : Использование функции обратного вызова (callback) при обучении модели.

**Пример использования**
```python
from src.callbacks.save import SaveModel

save_dir = "reports"
callbacks = [SaveModel(save_dir + "/model.pth")]
trainer = Trainer(
    pinn=pinn,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=5000,
    update_grid_every=100,
    callbacks=callbacks,
)
trainer.train()
```
## CallbacksOrganizer
    CLASS callbacks.callbacks_organizer.CallbacksOrganizer(self, callbacks: List[BaseCallback], mkdir: bool = True)
Класс для сортировки и хранения функций обратного вызова (callback), а также создания директории для сохранения файлов.

**Параметры**

- **callbacks** (List[BaseCallback]) : список функций обратного вызова (callback),
- **mkdir** (bool) : флаг создания директорий для функций обратного вызова с сохранением файлов.

**Пример использования**
```python
save_dir = "reports"
grid = src.callbacks.heatmap.Grid.from_pinn(pinn, 10001)
callbacks = [  
    src.callbacks.progress.TqdmBar('Epoch {epoch} lr={lr:.2e} Loss={loss_eq} Total={total_loss:.2e}'),  
    src.callbacks.curve.LearningRateCurve(save_dir, 500, log_scale=False),  
    src.callbacks.curve.LossCurve(save_dir, 100),  
    src.callbacks.curve.GridResidualCurve(save_dir, 100, grid=grid),  
    src.callbacks.heatmap.HeatmapPrediction(save_dir, 500, grid=grid)]
trainer = Trainer(
    pinn=pinn,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=5000,
    update_grid_every=100,
    callbacks_organizer=CallbacksOrganizer(callbacks))
trainer.train()
```

## EnsembleCallbacksOrganizer
	class callbacks.callbacks_organizer.EnsembleCallbacksOrganizer(self, callbacks: List[BaseCallback])

Класс для сортировки и хранения функций обратного вызова (callback) для использования в ансамбле. Класс позволяет использовать функции обратного вызова (callback) во всех моделях ансамбля.
**Параметры**

- **callbacks** (List[BaseCallback]) : список функций обратного вызова (callback).
