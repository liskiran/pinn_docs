#Нейронные сети


## Feed forward neural network
    CLASS: src.neural_network.FNN(layers_all: List[int])

Модель включает в себя целостный класс FNN, где задается количество нейронов. Архитектура FNN включает линейные слои, которые связаны между собой адаптивной функцией активации Sirena и функциями GeLU() [source to актив функции].В процессе обучения в методе forward модель обрабатывает входящие данные через каждый ленный слой для предсказания решения.

**Параметры**

- **layers_all** (List) - список нейронов по слоям для обучения.

> **_Важно:_**  Первое и последнее значение массива нейронов должно соответствовать геометрией задачи и количеством выводов.


**Методы**

    forward(x: torch.Tensor)

**Параметры**
        x (torch.Tensor) - входной тензор данных для прямого прохода модели

Выполняет прямой проход обучения модели. 


**Пример**


    model = FNN(layers_all=[2, 128, 128, 128, 1])


## Xavier feed forward neural network
    CLASS: src.neural_network.XavierFNN (layers_all: List[int], init_mode: str)

Модель включает в себя полноценный класс XavierFNN, в котором задается количество нейронов и метод инициализации весов ксавьера. Архитектура сети включает линейные слои, которые связаны между собой функциями активации Siren и GeLU [source функц актив].

**Параметры**

- **layers_all** (List) - список нейронов по слоям для обучения.
- **init_mode** (str) - метод инициализации весов ‘norm’/’uniform’ 

**Методы**

    forward(input: torch.Tensor)

Выполняет forward-pass модели. 

**Пример**


   model = XavierFNN (layers_all=[2, 128, 128, 128, 1], mode=’norm’)


    
## ResNet
    CLASS: src.neural_network.ResNet(layers_all: List[int], blocks: List[int], res_block: nn.Module, activation_function_array: List[nn.Module])

Модель включает в себя классы ResNet и LightResidualBlock. В самой модели ResNet задается количество линейных блоков и нейронов для каждого блока соответственно, более того при задании класса требуется передать вид линейного блока (LightResidualBlock) и функции активации, которые будут соединять линейные слои в блоках. Во время обучения происходит skip-connection, при котором пропускается часть слоев через блоки линейных слоев. 

**Параметры**

- **layers_all**(List) – список, содержащий количества нейроннов для каждого блока слоев Resnet.
> **_Важно:_**  первое и последнее значение массива нейронов должно соответствовать геометрией задачи и количеством выводов.
- **blocks**(List) – список, содержащий количество блоков для каждого количества нейронновю
> **_Важно:_**  При подаче None для каждого слоя будет использовать один блок.
- **res_block**(nn.Module) -бБлок ResNet, реализованный на базе ResidualBlockю
- **activation_function_array**(List) – список, содержащий функции активации, которые используются при обучении.

**Методы**

    forward(x: torch.Tensor)

Выполняет forward-pass модели. 

    __make_layers(res_block: nn.Module,count_blocks: int,in_features: int,out_features: int,activation: nn.Module, is_not_last: bool):

Конструирует блоки ResidualBlock одного размера и последующий линейнный слой новой размерности.

**Пример**


   model = ResNet (layers_all=[ 2, 5, 10, 15, 20, 25, 1], blocks = [1, 6, 5, 6, 2])

   
## LightResidualBlock
      CLASS: src.neural_network. LightResidualBlock (activation: nn.Module, features: int)

Архитектура блока в сети ResNet. На вход блок сети принимает входные и выходные значения для линейных слоев и функцию активации для связи блоков

**Параметры**

- **features**(int) – Размерность входных данных.
- **activation**(nn.Module) – функция активации


**Пример**


   layer_1 = LightResidualBlock(nn.GeLU(), 64)

   
## GELU
    CLASS: src.neural_network.activation_function.GELU(nn.Module)

Функция активации GeLU с ускорением расчетов в процессе обучения.

**Методы**
    forward(input: torch.Tensor)

Расчитывает функцию с оптимизацией torch.jit.script.


**Пример**

   gelu = GELU()


## Sine
    CLASS: src.neural_network.activation_function.Sine(nn.Module)

Адаптиваня функция активации Siren с ускорением расчетов в процессе обучения.

**Методы**
    forward(input: torch.Tensor)

Расчитывает функцию синуса с оптимизацией torch.jit.script.

**Пример**


   sine = Sine()
   

## DenseNet 
    CLASS: src.neural_network.DenseNet(layers_all: List[int], blocks: List[int], res_block: nn.Module, activation_function_array: List[nn.Module])

Архитектура модели включает в себя классы DenseNet и LightResidualBlock. В DenseNet задается количество линейных блоков и нейронов для каждого блока соответственно, более того при задании класса требуется передать вид линейного блока (LightResidualBlock) и функции активации, которые будут соединять линейные слои в блоках. Во время обучения происходит передача данных с каждого слоя на каждый, благодаря чему нейронная сеть получает больше информации о задаче.

**Параметры**

- **layers_all**(List) – список, содержащий количества нейроннов для каждого блока слоев Resnet.
> **_Важно:_**  первое и последнее значение массива нейронов должно соответствовать геометрией задачи и количеством выводов.
- **blocks**(List) – список, содержащий количество блоков для каждого количества нейронновю
> **_Важно:_**  При подаче None для каждого слоя будет использовать один блок.
- **res_block**(nn.Module) -бБлок ResNet, реализованный на базе ResidualBlockю
- **activation_function_array**(List) – список, содержащий функции активации, которые используются при обучении.

**Методы**

    forward(x: torch.Tensor)

Выполняет forward-pass модели. 

    __make_layers(res_block: nn.Module,count_blocks: int,in_features: int,out_features: int,activation: nn.Module, is_not_last: bool):

Конструирует блоки ResidualBlock одного размера и последующий линейнный слой новой размерности.


**Пример**

    model = DenseNet(layers_all=[ 2, 5, 10, 15, 20, 25, 1], blocks = [1, 6, 5, 6, 2])

