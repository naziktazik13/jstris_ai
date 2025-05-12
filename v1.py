from PIL import ImageGrab
import numpy as np
import time
from collections import deque

# Настройки области захвата (настройте под свой экран)
x, y, width, height = 580, 213, 250, 500
cols, rows = 10, 20
cell_width = width // cols
cell_height = height // rows

# Цвета Jstris
WHITE = (255, 255, 255)
GRAY = (106, 106, 106)
BACKGROUND = (0, 0, 0)  # Предполагаемый цвет фона

def get_game_matrix():
    """Захватывает экран и преобразует его в матрицу"""
    screenshot = ImageGrab.grab(bbox=(x, y, x + width, y + height))
    screenshot = screenshot.convert("RGB")
    
    matrix = []
    for row in range(rows):
        matrix_row = []
        for col in range(cols):
            cx = col * cell_width + cell_width // 2
            cy = row * cell_height + cell_height // 2
            pixel = screenshot.getpixel((cx, cy))
            
            if pixel == WHITE:
                matrix_row.append(1)  # Белый блок (активный или статичный)
            elif pixel == GRAY:
                matrix_row.append(2)  # Серый блок (только статичный)
            else:
                matrix_row.append(0)  # Пусто
        matrix.append(matrix_row)
    return matrix

def find_connected_components(matrix, target=1):
    """Находит все связанные компоненты заданного типа"""
    visited = set()
    components = []
    
    for r in range(rows):
        for c in range(cols):
            if matrix[r][c] == target and (r, c) not in visited:
                # Поиск в ширину
                queue = deque([(r, c)])
                visited.add((r, c))
                component = []
                
                while queue:
                    x, y = queue.popleft()
                    component.append((x, y))
                    
                    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < rows and 0 <= ny < cols and 
                            matrix[nx][ny] == target and 
                            (nx, ny) not in visited):
                            visited.add((nx, ny))
                            queue.append((nx, ny))
                
                components.append(component)
    return components

def identify_active_piece(current, previous):
    """Определяет активный тетромино с улучшенной логикой"""
    result = [[0 for _ in range(cols)] for _ in range(rows)]
    
    # Сначала помечаем все серые блоки как статичные
    for r in range(rows):
        for c in range(cols):
            if current[r][c] == 2:
                result[r][c] = 1
    
    # Находим все белые компоненты
    white_components = find_connected_components(current, 1)
    
    if not white_components:
        return result
    
    # Фильтруем компоненты по размеру (тетромино состоит из 4 блоков)
    candidates = [comp for comp in white_components if len(comp) == 4]
    
    if not candidates:
        # Если нет компонент из 4 блоков, берем самую большую
        candidates = [max(white_components, key=len)]
    
    # Выбираем самую высокую компоненту (активный тетромино обычно вверху)
    active_component = min(candidates, key=lambda comp: min(r for r, c in comp))
    
    # Помечаем активный тетромино
    for r, c in active_component:
        result[r][c] = 2
    
    # Помечаем остальные белые блоки как статичные
    for comp in white_components:
        if comp != active_component:
            for r, c in comp:
                result[r][c] = 1
    
    # Дополнительная проверка: если активный тетромино в самом низу, вероятно он стал статичным
    if any(r == rows-1 for r, c in active_component):
        for r, c in active_component:
            result[r][c] = 1
    
    return result

def print_matrix(matrix):
    """Красиво выводит матрицу"""
    symbols = {0: "□", 1: "■", 2: "▣"}
    for row in matrix:
        print(" ".join(symbols[cell] for cell in row))
    print("-" * 20)

def main():
    prev_matrix = None
    print("Запуск анализатора Jstris...")
    print("□ - пусто, ■ - статичный блок, ▣ - активный блок")
    print("-" * 40)
    
    try:
        while True:
            current = get_game_matrix()
            processed = identify_active_piece(current, prev_matrix)
            
            if prev_matrix is None or not np.array_equal(prev_matrix, processed):
                print("Обновление поля:")
                print_matrix(processed)
                prev_matrix = processed
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nОстановлено пользователем")
    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    main()