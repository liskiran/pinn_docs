# Нейронные сети


## Feed forward neural network
    CLASS: src.neural_network.FNN(layers_all: List[int])

Модель включает в себя целостный класс FNN, где задается количество нейронов. Архитектура FNN включает линейные слои, которые связаны между собой адаптивной функцией активации Sirena и функциями [GeLU](#gelu).В процессе обучения в методе forward модель обрабатывает входящие данные через каждый ленный слой для предсказания решения.

**Параметры**

- **layers_all** (List) - список нейронов по слоям для обучения.

> **_Важно:_**  Первое и последнее значение массива нейронов должно соответствовать геометрией задачи и количеством выводов.


**Методы**

    forward(x: torch.Tensor)

**Параметры**

    x (torch.Tensor) - входной тензор данных для прямого прохода модели

Выполняет прямой проход обучения модели. 


**Примеры использования:**

```python
from src.neural_network import FNN

# Определение входных и выходных нейронов 
input_size = 2

output_size = 1

# Определение слоев нейронов в модели
layers = [input_size, 128, 128, 128, output_size]


# Определение модели
model = FNN(layers_all=layers)


# Выполнение прямого прохода модели
output = model(input_tensor)

```


## Xavier feed forward neural network
    CLASS: src.neural_network.XavierFNN (layers_all: List[int], init_mode: str)

Модель включает в себя полноценный класс XavierFNN, в котором задается количество нейронов и метод инициализации весов ксавьера. Архитектура сети включает линейные слои, которые связаны между собой функциями активации [Siren](#sine) и [GeLU](#gelu).

**Параметры**

- **layers_all** (List) - список нейронов по слоям для обучения.
- **init_mode** (str) - метод инициализации весов ‘norm’/’uniform’ 

**Методы**

    forward(input: torch.Tensor)

**Параметры**

    input (torch.Tensor) - входной тензор данных для прямого прохода модели

Выполняет прямой проход обучения модели. 


   
**Примеры использования:**

```python
from src.neural_network import XavierFNN

# Определение входных и выходных нейронов 
input_size = 2

output_size = 1

# Определение слоев нейронов в модели
layers = [input_size, 128, 128, 128, output_size]

# Определение метода инициализации весов
mode= 'norm'

# Определение модели
model = XavierFNN (layers_all=layers, mode=mode)


# Выполнение прямого прохода модели
output = model(input_tensor)

```

    
## Residual neural network
    CLASS: src.neural_network.ResNet(layers_all: List[int], blocks: List[int], res_block: nn.Module, activation_function_array: List[nn.Module])

Модель включает в себя классы ResNet и LightResidualBlock. В самой модели ResNet задается количество линейных блоков и нейронов для каждого блока соответственно, более того при задании класса требуется передать вид линейного блока (LightResidualBlock) и функции активации, которые будут соединять линейные слои в блоках. Во время обучения происходит skip-connection, при котором пропускается часть слоев через блоки линейных слоев. 

**Параметры**

- **layers_all**(List) – список, содержащий количества нейроннов для каждого блока слоев Resnet.
> **_Важно:_**  первое и последнее значение массива нейронов должно соответствовать геометрией задачи и количеством выводов.
- **blocks**(List) – список, содержащий количество блоков для каждого количества нейронновю
> **_Важно:_**  При подаче None для каждого слоя будет использовать один блок.
- **res_block**(nn.Module) - блок ResNet, реализованный на базе ResidualBlock.
- **activation_function_array**(List) – список, содержащий функции активации, которые используются при обучении.

**Методы**

    forward(x: torch.Tensor)

**Параметры**

    x (torch.Tensor) - входной тензор данных для прямого прохода модели

Выполняет прямой проход обучения модели. 


    __make_layers(res_block: nn.Module,count_blocks: int,in_features: int,out_features: int,activation: nn.Module, is_not_last: bool):

**Параметры**
   - res_block (nn.Module) - блок ResNet, реализованный на базе ResidualBlock
   - count_blocks (int) - колличество блоков ResNet
   - in_features (int) - колличество входных нейронов в слои блока ResNet
   - out_features (int) - колличество выходных нейронов в слои блока ResNet
   - activation (nn.Module) - функция активации, соеденяющаяя линейнные слои
   - is_not_last (bool) - ограничение, показывающее, что следующего блока ResNet нет
    
Конструирует блоки ResidualBlock одного размера и последующий линейнный слой новой размерности.



**Примеры использования:**

```python
from src.neural_network import ResNet

# Определение входных и выходных нейронов 
input_size = 2

output_size = 1

# Определение слоев нейронов в модели
layers = [input_size, 5, 10, 15, 20, 25, output_size]

# Определение слоев нейронов в модели
blocks = [1, 6, 5, 6, 2]


# Определение модели
model = ResNet(layers_all=layers, blocks = blocks)

# Выполнение прямого прохода модели
output = model(input_tensor)

```

## LightResidualBlock
      CLASS: src.neural_network. LightResidualBlock (activation: nn.Module, features: int)

Архитектура блока в Residual neural network. На вход блок сети принимает входные и выходные значения для линейных слоев и функцию активации для связи блоков

**Параметры**

- **features**(int) – Размерность входных данных.
- **activation**(nn.Module) – функция активации

**Методы**

    forward(x: torch.Tensor)

**Параметры**

    x (torch.Tensor) - входной тензор данных для прямого прохода модели

Выполняет прямой проход обучения блока модели ResNet.

**Пример**

   layer = LightResidualBlock(nn.GeLU(), 64)

**Примеры использования:**

```python
from src.neural_network import LightResidualBlock
from src.neural_network.activation_function import GeLU
input_neuron = 64

# Определение функции активации
gelu = GeLU()

# Определение блока модели 
layer = LightResidualBlock(features = input_neuron, activation = gelu)

```

## DenseNet 
    CLASS: src.neural_network.DenseNet(layers_all: List[int], blocks: List[int], res_block: nn.Module, activation_function_array: List[nn.Module])

Архитектура модели включает в себя классы DenseNet и LightResidualBlock. В DenseNet задается количество линейных блоков и нейронов для каждого блока соответственно, более того при задании класса требуется передать вид линейного блока (LightResidualBlock) и функции активации, которые будут соединять линейные слои в блоках. Во время обучения происходит передача данных с каждого слоя на каждый, благодаря чему нейронная сеть получает больше информации о задаче.



## GELU
    CLASS: src.neural_network.activation_function.GELU(nn.Module)

Функция активации GeLU с ускорением расчетов в процессе обучения.

**Методы**
    forward(input: torch.Tensor)

**Параметры**
    input (torch.Tensor) - входной тензор данных для прямого прохода функции
    
Выполняет прямой проход функции активации во время обучения модели с оптимизацией torch.jit.script.

**Пример**

   gelu = GELU()

**Примеры использования:**

```python
from src.neural_network.activation_function import GeLU

# Определение функции активации
gelu = GeLU()

# Выполнение прямого прохода 
output = gelu(input_tensor)

```

## Sine
    CLASS: src.neural_network.activation_function.Sine(nn.Module)

Адаптиваня функция активации Siren с ускорением расчетов в процессе обучения.

**Методы**
    forward(input: torch.Tensor)

**Параметры**
    input (torch.Tensor) - входной тензор данных для прямого прохода функции
    
Выполняет прямой проход адаптивной функции синуса во время обучения модели с оптимизацией torch.jit.script.


**Пример**

   sine = Sine()
   
**Примеры использования:**

```python
from src.neural_network.activation_function import Sine

# Определение функции активации
sine = Sine()

# Выполнение прямого прохода 
output = sine(input_tensor)

```
