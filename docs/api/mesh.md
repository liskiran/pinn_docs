# Работа с файлами **.msh**

На данный момент PINN-ADT поддерживает файлы MESH следующийх типов: fluent, TGrid.

## GridReader

```python
class grid_reader.GridReader()
```

Класс **GridReader** считывает MESH файл и преобразует его в тип данных **Grid**

### Методы:

```python
read(filename: str)
```

Считывает файл лежащий в пути **filename** и преобразует его в тип данных **Grid**

## Grid

```python
class grid_reader.Grid()
```

Класс **Grid** представляет собой класс данных, в котором хранится вся необходимая информация для работы PINN-ADT.

### Методы:

```python
plot_nd(plot_normals: bool, normals_size: float) 
```

Отображает **Grid** в виде html страницы.

#### Параметры:

- **plot_normals** (bool, по умолчанию True) – отображать ли нормали, направленные внутрь области.
- **normals_size** (float, по умолчанию 10.0) – длинна векторов нормали. Если **plot_normals = False**, то параметр ни
  на что не влияет.

## Пример использования
```python
from grid_reader import GridReader

grid_file = GridReader().read("./mesh.msh")

grid_file.plot_nd()
```

