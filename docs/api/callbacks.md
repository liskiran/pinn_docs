## Grid
    CLASS callbacks.heatmap.Grid(low: np.array, high: np.array, n_points: Union[Sequence[int], int])

Класс, позволяющий создать сетку точек в указанной области. Сетку можно использовать для построения тепловых карт и расчета значения функций на структурированном наборе точек

**Параметры**

- **low** (np.array) – массив со значениями нижних границ для каждой оси,
- **high** (np.array) – np.array со значениями верхних границ для каждой оси,
- **n_points** (Union[Sequence[int], int]) — массив со значениями количества точек по каждой оси / общее количество точек на весь Grid.

**Методы**

- **from_pinn(cls, pinn: PINN, n_points: Union[Sequence[int], int])** : Создает объект Grid на основе PINN и указанного количества точек pinn
- **from_condition(cls, condition, n_points: Union[Sequence[int], int])** : Создает объект Grid на основе условий задачи и указанного количества точек.
- **smart_volume(low, high, total_points: int) -> Sequence[int]** : Статический метод для расчета количества точек по каждой оси на основе указанного числа точек.

**Примеры использования**

Построение сетки с указанными верхней и нижней границами и количеством точек по осям:
```python
grid = Grid(low=[0, 0], high=[1, 1], n_points=[100, 100])
```
Построение сетки с указанными верхней и нижней границами и общим количеством точек:
```python
grid = Grid(low=[0, 0], high=[1, 1], n_points=10001)
```
Построение сетки на основе экземпляра класса PINN и указанного количества точек:
```python
conditions, input_dim, output_dim = problem_2D1C_heat_equation()
model = FNN(layers_all=[input_dim, 128, 128, 128, output_dim])
pinn = PINN(model=model, conditions=conditions)
grid = Grid.from_pinn(pinn, 80001)
```
Построение сетки на основе условий задачи и указанного количества точек:
```python
conditions, input_dim, output_dim = problem_2D1C_heat_equation()
grid = Grid.from_condition(conditions, 10001)
```


  

## BaseSave
    CLASS callbacks.heatmap.BaseSave(self, save_dir: str, period: int, save_mode: str = 'html')
    
Абстрактный класс для наследования класса BaseHeatmap

**Параметры**

- **save_dir** (str) – директория для сохранения графика,
- **period** (int) – период сохранения графиков,
- **save_mode** (str) — режим сохранения (Режимы сохранения: “html” - сохраняет каждую тепловую карту в указанной директории в формате html; “png” - сохраняет тепловые карты в указанной директории в формате png; “pt” - сохраняет точки, по которым можно построить данную тепловую карту; “show” - открывает каждую тепловую карту в браузере в интерактивном режиме, далее ее можно сохранить вручную).

**Методы**

- **save_fig(self, fig, file_name: str)** : Сохраняет или отрисовывает график в одном из режимов.
- **save_pt(self, fig, file: str)** : Метод для сохранения точек графика.
  
## BaseHeatmap
    CLASS callbacks.heatmap.BaseHeatmap(self, save_dir: str, grid: Grid, period: int = 500, save_mode: str = 'html'):
    
Абстрактный класс для наследования другими классами тепловых карт.

**Параметры**

- **save_dir** (str) – директория для сохранения графика,
- **grid** (str) – объект класса Grid,
- **period** (int) – период сохранения графиков,
- **save_mode** (str) — режим сохранения (Режимы сохранения: “html” - сохраняет каждую тепловую карту в указанной директории в формате html; “png” - сохраняет тепловые карты в указанной директории в формате png; “pt” - сохраняет точки, по которым можно построить данную тепловую карту; “show” - открывает каждую тепловую карту в браузере в интерактивном режиме, далее ее можно сохранить вручную).

**Методы**

- **draw(self, values: torch.Tensor, plot_name: str, file_name: str = None)** : В зависимости от размерности вызывают одну из функций для построения тепловой карты.
- **dict_data(self, fig)** : Метод для сохранения точек графика.
- **draw_3D(self, values: torch.Tensor, plot_name: str, file_name: str = None)** : Строит трехмерную тепловую карту и сохраняет ее в выбранном формате.
- **draw_2D(self, values: torch.Tensor, plot_name: str, file_name: str = None, min = None, max=None)** : Строит двумерную тепловую карту и сохраняет ее в выбранном формате.
- **draw_1D(self, values: torch.Tensor, plot_name: str, file_name: str = None)** : Строит одномерную тепловую карту и сохраняет ее в выбранном формате.
## HeatmapError

    CLASS callbacks.heatmap.HeatmapError(self, save_dir: str, grid: Grid, solution: Callable[[torch.Tensor], torch.Tensor], period: int = 500,
                 save_mode: str = 'html')
    
Функция обратного вызова(callback) для создания тепловых карт ошибок во время обучения модели.

**Параметры**

- **save_dir** (str) – директория для сохранения графика,
- **grid** (str) – объект класса Grid,
- **solution** — функция точного решения,
- **period** (int) – период сохранения графиков,
- **save_mode** (str) — режим сохранения (Режимы сохранения: “html” - сохраняет каждую тепловую карту в указанной директории в формате html; “png” - сохраняет тепловые карты в указанной директории в формате png; “pt” - сохраняет точки, по которым можно построить данную тепловую карту; “show” - открывает каждую тепловую карту в браузере в интерактивном режиме, далее ее можно сохранить вручную).

**Методы**

- **__call__(self, trainer: Trainer))** : Использование функции обратного вызова(callback) при обучении модели.

## HeatmapPrediction

    CLASS callbacks.heatmap.HeatmapPrediction(self, save_dir: str, grid: Grid, period: int = 500, save_mode: str = 'html')
    
Функция обратного вызова(callback) для создания тепловых карт решения, полученного моделью.

**Параметры**

- **save_dir** (str) – директория для сохранения графика,
- **grid** (str) – объект класса Grid,
- **period** (int) – период сохранения графиков,
- **save_mode** (str) — режим сохранения (Режимы сохранения: “html” - сохраняет каждую тепловую карту в указанной директории в формате html; “png” - сохраняет тепловые карты в указанной директории в формате png; “pt” - сохраняет точки, по которым можно построить данную тепловую карту; “show” - открывает каждую тепловую карту в браузере в интерактивном режиме, далее ее можно сохранить вручную).

**Методы**

- **__call__(self, trainer: Trainer))** : Использование функции обратного вызова(callback) при обучении модели.
## PlotHeatmapSolution

    CLASS callbacks.heatmap.PlotHeatmapSolution(self, save_dir: str, grid: Grid, solution: Callable[[torch.Tensor], torch.Tensor],
                 save_mode: str = 'html')
    
Класс для построения графика точного решения. Это не функция обратного вызова(callback)!

**Параметры**

- **save_dir** (str) – директория для сохранения графика,
- **grid** (str) – объект класса Grid,
- **solution** — функция точного решения,
- **save_mode** (str) — режим сохранения (Режимы сохранения: “html” - сохраняет каждую тепловую карту в указанной директории в формате html; “png” - сохраняет тепловые карты в указанной директории в формате png; “pt” - сохраняет точки, по которым можно построить данную тепловую карту; “show” - открывает каждую тепловую карту в браузере в интерактивном режиме, далее ее можно сохранить вручную).

## BasicCurve
    CLASS callbacks.curve.BasicCurve(self, save_dir: str, period: int = 500, save_mode: str = 'html', log_scale: bool = True)
Абстрактный класс для наследования другими классами тепловых карт.
**Параметры**

- **save_dir** (str) – директория для сохранения графика,
- **period** (int) – период сохранения графиков,
- **save_mode** (str) — режим сохранения (Режимы сохранения: “html” - сохраняет каждую кривую в указанной директории в формате html; “png” - сохраняет тепловые карты в указанной директории в формате png; “pt” - сохраняет точки, по которым можно построить данную тепловую карту; “show” - открывает каждую тепловую карту в браузере в интерактивном режиме, далее ее можно сохранить вручную),
- **log_scale**(bool) — флаг, определяющий будет ли ось OY логарифмирована.

**Методы**

- **draw(self, values: Sequence, v_names: [str], coord: np.ndarray, plot_title: str, file_name: str = None)** : Строит и сохраняет кривую.
- **dict_data(fig)** : Сохраняет данные графика в виде словаря.
- **init_curve_names(self, conditions)** : Нзначает названия кривым в зависимости от conditions.

## GridResidualCurve
    CLASS callbacks.curve.GridResidualCurve(self, save_dir: str, grid: Grid, period=100, save_mode='html', log_scale: bool = True)
Функция обратного вызова(callback) для построения кривых обучения при помощи сетки.

**Параметры**

- **save_dir** (str) – директория для сохранения графика,
- **grid**(Grid) — Объект класса Grid,
- **period** (int) – период сохранения графиков,
- **save_mode** (str) — режим сохранения (Режимы сохранения: “html” - сохраняет каждую кривую в указанной директории в формате html; “png” - сохраняет тепловые карты в указанной директории в формате png; “pt” - сохраняет точки, по которым можно построить данную тепловую карту; “show” - открывает каждую тепловую карту в браузере в интерактивном режиме, далее ее можно сохранить вручную),
- **log_scale** (bool) — флаг, определяющий будет ли ось OY логарифмирована.

**Методы**

- **__call__(self, trainer: Trainer) -> None** : Использование функции обратного вызова(callback) при обучении модели.

## TrainingCurve
    CLASS callbacks.curve.TrainingCurve(self, save_dir: str, period=100, save_mode='html', log_scale: bool = True)
Функция обратного вызова(callback) для построения кривых обучения по лоссам модели.

**Параметры**

- **save_dir** (str) – директория для сохранения графика,
- **period** (int) – период сохранения графиков,
- **save_mode** (str) — режим сохранения (Режимы сохранения: “html” - сохраняет каждую кривую в указанной директории в формате html; “png” - сохраняет тепловые карты в указанной директории в формате png; “pt” - сохраняет точки, по которым можно построить данную тепловую карту; “show” - открывает каждую тепловую карту в браузере в интерактивном режиме, далее ее можно сохранить вручную),
- **log_scale** (bool) — флаг, определяющий будет ли ось OY логарифмирована.

**Методы**

- **__call__(self, trainer: Trainer) -> None** : Использование функции обратного вызова(callback) при обучении модели.
## LearningRateCurve
    CLASS callbacks.curve.LearningRateCurve(self, save_dir: str, period=100, save_mode='html', log_scale: bool = True)
Функция обратного вызова(callback) для построения кривой learning rate.

**Параметры**

- **save_dir** (str) – директория для сохранения графика,
- **period** (int) – период сохранения графиков,
- **save_mode** (str) — режим сохранения (Режимы сохранения: “html” - сохраняет каждую кривую в указанной директории в формате html; “png” - сохраняет тепловые карты в указанной директории в формате png; “pt” - сохраняет точки, по которым можно построить данную тепловую карту; “show” - открывает каждую тепловую карту в браузере в интерактивном режиме, далее ее можно сохранить вручную),
- **log_scale** (bool) — флаг, определяющий будет ли ось OY логарифмирована.

**Методы**

- **__call__(self, trainer: Trainer) -> None** : Использование функции обратного вызова(callback) при обучении модели.
## ErrorCurve
    CLASS callbacks.curve.ErrorCurve(self, save_dir: str, solution: Callable[[torch.Tensor], torch.Tensor], period=100, save_mode='html',
                 log_scale: bool = True)
Функция обратного вызова(callback) для построения кривой ошибки.

**Параметры**

- **save_dir** (str) – директория для сохранения графика,
- **solution** (Callable) — функция точного решения,
- **period** (int) – период сохранения графиков,
- **save_mode** (str) — режим сохранения (Режимы сохранения: “html” - сохраняет каждую кривую в указанной директории в формате html; “png” - сохраняет тепловые карты в указанной директории в формате png; “pt” - сохраняет точки, по которым можно построить данную тепловую карту; “show” - открывает каждую тепловую карту в браузере в интерактивном режиме, далее ее можно сохранить вручную),
- **log_scale** (bool) — флаг, определяющий будет ли ось OY логарифмирована.

**Методы**

- **__call__(self, trainer: Trainer) -> None** : Использование функции обратного вызова(callback) при обучении модели.
## ProgressBar
    CLASS callbacks.progress.ProgressBar(self, template: str, period: int = 10)
Функция обратного вызова(callback) для обновления Progress Bar.

**Параметры**

- **template** (str) – шаблон строки, выводимой в консоль,
- **period** (int) — периодичность использования функции обратного вызова(callback).

**Методы**

- **__call__(self, trainer: Trainer) -> None** : Использование функции обратного вызова(callback) при обучении модели.
- **make_message(self, trainer: Trainer) -> str** : Создание строки, выводимой в консоль.
## TqdmBar
    CLASS callbacks.progress.TqdmBar(self, template: str, period: int = 10)
Функция обратного вызова(callback) для обновления Tqdm progress bar.

**Параметры**

- **template** (str) – шаблон строки, выводимой в консоль,
- **period** (int) — периодичность использования функции обратного вызова(callback).

**Методы**

- **__call__(self, trainer: Trainer) -> None** : Использование функции обратного вызова(callback) при обучении модели.
## SaveModel
    CLASS callbacks.save.SaveModel(self, save_path: str, period: int = 1000)
Функция обратного вызова(callback) для сохранения модели.

**Параметры**

- **save_path** (str) – путь для сохранения файла(включая название),
- **period** (int) — периодичность использования функции обратного вызова(callback).

**Методы**

- **__call__(self, trainer: Trainer) -> None** : Использование функции обратного вызова(callback) при обучении модели.
## CallbacksOrganizer
    CLASS callbacks.callbacks_organizer.CallbacksOrganizer(self, callbacks: List[BaseCallback])
Класс для сортировки функций обратного вызова(callback).

**Параметры**

- **callbacks** (List[BaseCallback]) – список функций обратного вызова(callback).

