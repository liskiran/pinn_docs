# Generation

## BaseGeometry
    class BaseGenerator(ABC)

Абстрактный класс для наследования классов, отвечающих за генерацию точек в области

**Методы**

 - **generate(self, geometry: BaseGeometry, condition: Condition, model)** : Абстрактный метод для генерации точек


## UniformGeneratorRect
    class UniformGeneratorRect(self, n_points, method)

Класс неадаптивных алгоритмов генерации точек для прямоугольных и линейных областей. 

**Параметры** 

- **n_points** (int) : количество точек задаваемое в области
- **method** (str) : название метода генерации

**Методы**

- **generate(self, geometry: RectangleArea, condition: Condition, model)** -> (torch.tensor, torch.tensor) : общий метод генерации точек. Не содержит в себе реализацию метода, выполняет функцию вызова нужной реализации на основе параметров генератора.

- **generate_uniform(self, geometry: RectangleArea)** -> (torch.tensor, torch.tensor) : реализует равномерную случайную генерацию точек в области.


**Пример**

```python
generator_domain = UniformGeneratorRect(n_points=5000,
                                            method='uniform')
```

## AdaptiveGeneratorRect
    class AdaptiveGeneratorRect(self, n_points, method: str, power_coeff=3, add_coeff=1, density_rec_points_num=None,
                 add_points=None, n_points_up_bnd=None):

Класс адаптивных алгоритмов генерации точек для прямоугольных и линейных областей. 

Адаптивные  стратегии строят сетку на основе информации об ошибке нейронной сети. 

При обновлении точек происходит построение поля невязки $\xi (x) = |f(x, \hat u(x))|$. Далее применяется преобразование
$p(x) \propto \frac{\xi^k (x)}{E[\xi^k (x)]} + c$, позволяющее регулировать степень адаптивности алгоритма.


**Параметры** 

- **n_points** (int) : количество точек задаваемое в области
- **method** (str) : название метода генерации
- **power_coeff** (int) : степенной коэффициент $k$
- **add_coeff** (int) : коэффициент $c$
- **density_rec_points_num** (int) : количество точек по которым строится поле невязок
- **add_points\*** (int) : количество точек, добавляемых при обновлении точек. 
- **n_points_up_bnd\*** (int) : максимальное количество точек, которое может быть достигнуто при последовательном добавлении точек. 

\*  параметры используются в алгоритмах (RAR-D, RAR-G) с постепенным добавлением точек.

**Методы**

- **generate(self, geometry: RectangleArea, condition: Condition, model)** -> (torch.tensor, torch.tensor) : общий метод генерации точек. Не содержит в себе реализацию метода генерации, выполняет функцию вызова нужной реализации, на основе параметров генератора.

- **generate_RAD(self, geometry: RectangleArea, condition: Condition, model)** -> (torch.tensor, torch.tensor) : генерирует точки за счёт выбора точек на основе невязки и вероятностного преобразования.

- **generate_RAG(self, geometry: RectangleArea, condition: Condition, model)** -> (torch.tensor, torch.tensor) : генерирует точки за счет выбора точек с наибольшей невязкой. Множество точек обновляется полностью.

- **generate_RAR-D(self, geometry: RectangleArea, condition: Condition, model)** -> (torch.tensor, torch.tensor) : генерирует точки за счёт постепенного добавления точек на основе вероятностоного преобразования. При первой генерации точек, они распределены равномерно. Далее, при каждой генерации в текущее множество точек добавляются add_points точек на основе и невязки и вероятностного преобразования.

- **generate_RAR-G(self, geometry: RectangleArea, condition: Condition, model)** -> (torch.tensor, torch.tensor) :  генерирует точки за счёт постепенного добавления точек с наибольшей невязкой. При первой генерации точек, они распределены равномерно. Далее, при каждой генерации в текущее множество точек добавляются add_points точек c наибольшей невязкой.

- **calc_error_field(self, geometry: RectangleArea, condition: Condition, model)** -> (torch.tensor, torch.tensor) : генерирует множество точек, после чего считает невязку на нём. 

- **sample_from_density(self, errors, points_num)** -> (torch.tensor) : применяет вероятностное преобразование и генерирует из него подвыборку 

**Пример**

```python
generator_domain = AdaptiveGeneratorRect(2048, 'RAG', power_coeff=3, add_coeff=0,
                                          density_rec_points_num=8192, add_points=32)
```



## UniformGeneratorMesh
    class UniformGeneratorMesh(self, n_points, method)

Класс неадаптивных алгоритмов генерации точек обучения на основе mesh файлов. 

**Параметры** 

- **n_points** (int) : количество точек задаваемое в области
- **method** (str) : название метода генерации

**Методы**

- **generate(self, geometry: RectangleArea, condition: Condition, model)** -> (torch.tensor, torch.tensor) : общий метод генерации точек. Не содержит в себе реализацию метода, выполняет функцию вызова нужной реализации, на основе параметров генератора.

- **uniform_sample(self, geometry: MeshArea)** -> (torch.tensor, torch.tensor) : генерирует подвыборку точек из множества точек, находящихся в mesh файле. 

**Пример**

```python
generator_domain = UniformGeneratorMesh(n_points = 15000, method = 'uniform_sample')
```