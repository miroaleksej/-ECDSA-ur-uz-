# Hypercube Analysis of ECDSA Signatures

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![SageMath](https://img.shields.io/badge/SageMath-9.0%2B-orange)](https://www.sagemath.org/)

Этот репозиторий содержит инновационные исследования в области криптоанализа ECDSA через построение и анализ **гиперкуба параметров (ur, uz)**. Мы раскрываем скрытые топологические структуры цифровых подписей, разрабатываем методы обнаружения уязвимостей и создаем инструменты для практического криптоанализа.

## 🔍 Основные открытия

1. **Биекция гиперкуба**: Установлено взаимно-однозначное соответствие между валидными подписями ECDSA и точками в пространстве `ℤₙ × ℤₙ`:
   ```math
   (u_r, u_z) \leftrightarrow (r, s, z)
   ```
   где:
   - `uᵣ = r·s⁻¹ mod n`
   - `u_z = z·s⁻¹ mod n`

2. **Топология коллизий**: Линии коллизий в гиперкубе описываются уравнениями:
   ```math
   \Delta u_z \equiv -d \cdot \Delta u_r \mod n
   ```
   где `d` - приватный ключ. Эти линии образуют регулярную структуру, зависящую от `d`.

3. **Голографический принцип**: Локальные микроучастки гиперкуба (размером `O(log² n)`) сохраняют глобальные топологические свойства и позволяют восстанавливать информацию о приватном ключе.

## 🛠 Установка и использование

### Зависимости
- Python 3.8+
- SageMath
- Библиотеки: `numpy`, `scipy`, `matplotlib`, `qiskit` (для квантовых алгоритмов), `sklearn`

```bash
pip install -r requirements.txt
```

### Пример: Построение микроучастка гиперкуба
```python
from hypercube_tile import map_hypercube_tile, analyze_tile_topology
from sage.all import EllipticCurve, GF

# Инициализация кривой (secp256k1)
E = EllipticCurve(GF(0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F), [0,7])
n = E.order()
G = E.gens()[0]
Q = d * G  # Публичный ключ

# Построение микроучастка
tile, collision_map = map_hypercube_tile(E, Q, seed_point=(0, 0), tile_size=1024)

# Анализ топологии
topology = analyze_tile_topology(tile)
print(f"Фрактальная размерность: {topology['fractal_dimension']}")
```

### Пример: Генерация цифрового отпечатка ключа
```python
from key_fingerprint import generate_key_fingerprint

fingerprint = generate_key_fingerprint(Q)
print(f"Топологический отпечаток: {fingerprint}")
```

## 📊 Визуализации

![Микроучастки гиперкуба](https://i.imgur.com/hypercube_tiles.png)
*Примеры микроучастков для secp256k1. Цвета кодируют значение `r` подписи, структура линий отражает приватный ключ `d`.*

## 🔬 Ключевые методы

1. **Адаптивный поиск коллизий**:
   ```python
   from collision_detection import adaptive_collision_search
   collisions = adaptive_collision_search(Q)
   ```

2. **Топологический анализ**:
   - Фрактальная размерность
   - Спектральный анализ (Фурье)
   - Детектирование сингулярностей

3. **Квантово-вдохновленные алгоритмы**:
   - VQE для поиска коллизий
   - Квантовая выборка микроучастков

## 📚 Теоретическая база

### Основные теоремы
1. **Биекция подписей**:
   ```math
   \forall (r,s,z)_{\text{валид}} \exists! (u_r, u_z): 
   \begin{cases} 
   u_r = r \cdot s^{-1} \mod n \\
   u_z = z \cdot s^{-1} \mod n 
   \end{cases}
   ```

2. **Теорема об индивидуальности**:
   Микроучасток размера `L > log²(n)` однозначно определяет глобальную структуру гиперкуба.

### Уравнение состояния гиперкуба
```math
P = \frac{\hbar c}{2\pi} \rho^3 + T \cdot s \cdot \exp(-\beta \phi) - \frac{\kappa}{8\pi G} \|\Gamma\|^2
```
где:
- `P` - давление безопасности
- `ρ` - плотность уязвимостей
- `T` - "температура" системы (хаотичность ГПСЧ)


## 🤝 Как внести вклад
1. Тестирование на других кривых (P-256, Curve25519)
2. Оптимизация алгоритмов поиска коллизий
3. Разработка новых методов визуализации

> "Гиперкуб параметров — это карта вселенной ECDSA, где каждая точка является звездой, а линии коллизий — космическими дорогами, ведущими к секретам." — Криптографическая онтология

## 📄 Лицензия
Проект распространяется под лицензией MIT. Подробнее см. [LICENSE](LICENSE).
