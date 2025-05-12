from PIL import ImageGrab
import numpy as np
import time
from collections import deque
import keyboard
import threading

# Настройки области захвата (настройте под свой экран)
x, y, width, height = 580, 213, 250, 500
cols, rows = 10, 20
cell_width = width // cols
cell_height = height // rows

# Цвета Jstris
WHITE = (255, 255, 255)  # Активный блок
GRAY = (106, 106, 106)   # Статичный блок
BGRAY = (153, 153, 153)  # Вариант серого блока
SHADOW = (127, 127, 127) # Тень активного блока
BACKGROUND = (0, 0, 0)   # Фон

# Управление AI  
ai_enabled = True
stop_flag = False 
piece_detected_frames = 1
MIN_FRAMES_TO_START = 1 

# Словарь известных фигур Тетриса и их форм
TETROMINO_SHAPES = {
    'I': [(0, 0), (0, 1), (0, 2), (0, 3)],  # I-блок (палка)
    'O': [(0, 0), (0, 1), (1, 0), (1, 1)],  # O-блок (квадрат)
    'T': [(0, 1), (1, 0), (1, 1), (1, 2)],  # T-блок
    'S': [(0, 1), (0, 2), (1, 0), (1, 1)],  # S-блок
    'Z': [(0, 0), (0, 1), (1, 1), (1, 2)],  # Z-блок
    'J': [(0, 0), (1, 0), (1, 1), (1, 2)],  # J-блок
    'L': [(0, 2), (1, 0), (1, 1), (1, 2)]   # L-блок
}

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
                matrix_row.append(1)  # Белый блок (активный)
            elif pixel == GRAY or pixel == BGRAY:
                matrix_row.append(2)  # Серый блок (статичный)
            elif pixel == SHADOW:
                matrix_row.append(3)  # Тень активного блока
            else:
                matrix_row.append(0)  # Пусто
        matrix.append(matrix_row)
    return matrix

def identify_active_piece_with_shadow(matrix):
    """
    Определяет активный тетромино используя тень как ориентир.
    Правила:
    1. Все блоки НАД тенью и до верха поля считаются активными
    2. Блоки ПОД тенью считаются статичными
    3. Сама тень не является частью активного блока
    """
    result = [[0 for _ in range(cols)] for _ in range(rows)]
    
    # Сначала находим все позиции тени
    shadow_positions = []
    for r in range(rows):
        for c in range(cols):
            if matrix[r][c] == 3:  # Тень
                shadow_positions.append((r, c))
    
    # Если тень не найдена, используем старый метод определения активных блоков
    if not shadow_positions:
        for r in range(rows):
            for c in range(cols):
                if matrix[r][c] == 2:  # Статичный блок
                    result[r][c] = 1
                elif matrix[r][c] == 1:  # Возможно активный блок
                    # Обрабатываем как активный, если не можем определить иначе
                    result[r][c] = 2
        return result
    
    # Определяем область влияния тени - колонки, в которых есть тень
    shadow_columns = set(c for _, c in shadow_positions)
    
    # Находим самую высокую строку тени для каждого столбца
    shadow_highest_row = {}
    for r, c in shadow_positions:
        if c not in shadow_highest_row or r < shadow_highest_row[c]:
            shadow_highest_row[c] = r
    
    # Обрабатываем каждую клетку
    for r in range(rows):
        for c in range(cols):
            # Пропускаем пустые клетки
            if matrix[r][c] == 0:
                continue
                
            # Сама тень не является блоком активного тетромино
            if matrix[r][c] == 3:
                continue
                
            # Если это серый блок, он всегда статичный
            if matrix[r][c] == 2:
                result[r][c] = 1
                continue
                
            # Если это столбец с тенью и блок находится выше тени
            if c in shadow_columns and r < shadow_highest_row[c]:
                result[r][c] = 2  # Активный блок
            else:
                result[r][c] = 1  # Статичный блок
    
    return result

def extract_piece_shape(matrix):
    """Извлекает форму активного тетромино из матрицы"""
    piece_cells = []
    
    for r in range(rows):
        for c in range(cols):
            if matrix[r][c] == 2:  # Активный блок
                piece_cells.append((r, c))
    
    if not piece_cells:
        return None
    
    # Нормализуем координаты для удобства
    min_r = min(r for r, c in piece_cells)
    min_c = min(c for r, c in piece_cells)
    
    normalized_piece = [(r - min_r, c - min_c) for r, c in piece_cells]
    return normalized_piece, min_r, min_c

def extract_shadow_position(matrix):
    """Извлекает позицию тени из матрицы"""
    shadow_cells = []
    
    for r in range(rows):
        for c in range(cols):
            if matrix[r][c] == 3:  # Тень
                shadow_cells.append((r, c))
    
    if not shadow_cells:
        return None
    
    # Нормализуем координаты для удобства
    min_r = min(r for r, c in shadow_cells)
    min_c = min(c for r, c in shadow_cells)
    
    normalized_shadow = [(r - min_r, c - min_c) for r, c in shadow_cells]
    return normalized_shadow, min_r, min_c

def identify_tetromino_from_shadow(shadow_cells):
    """
    Определяет тип тетромино по форме тени
    Возвращает нормализованную форму полного тетромино
    """
    if not shadow_cells:
        return None
    
    # Нормализуем тень
    normalized_shadow = set(shadow_cells[0])
    
    # Перебираем все возможные тетромино
    for shape_name, shape in TETROMINO_SHAPES.items():
        # Генерируем все возможные повороты для сравнения
        rotations = get_all_possible_rotations(shape)
        for rotation in rotations:
            rotation_set = set(rotation)
            # Если форма тени совпадает с текущей формой тетромино
            if len(rotation_set) == len(normalized_shadow) and all(cell in rotation_set for cell in normalized_shadow):
                return rotation
    
    # Если не удалось определить форму, возвращаем саму тень как форму
    return shadow_cells[0]

def get_static_board(matrix):
    """Извлекает статичную часть доски (без активной фигуры)"""
    board = [[0 for _ in range(cols)] for _ in range(rows)]
    
    for r in range(rows):
        for c in range(cols):
            if matrix[r][c] == 1:  # Только статичные блоки
                board[r][c] = 1
    
    return board

def get_all_possible_rotations(piece_shape):
    """Возвращает все возможные повороты фигуры"""
    if not piece_shape:
        return []
    
    rotations = []
    current_rotation = piece_shape
    
    # Для симметрии, добавляем до 4 поворотов (0°, 90°, 180°, 270°)
    for _ in range(4):
        rotations.append(current_rotation)
        # Поворот на 90° по часовой стрелке
        current_rotation = [(c, -r) for r, c in current_rotation]
        # Нормализация после поворота
        min_r = min(r for r, _ in current_rotation)
        min_c = min(c for _, c in current_rotation)
        current_rotation = [(r - min_r, c - min_c) for r, c in current_rotation]
        
        # Проверка, есть ли уже такой поворот
        if current_rotation in rotations:
            break
    
    return rotations

def simulate_drop(board, piece, col_offset):
    """Симулирует падение фигуры в указанную позицию"""
    if not piece:
        return None
    
    # Создаем копию доски
    new_board = [row[:] for row in board]
    
    # Начинаем с верхней строки и опускаем фигуру до столкновения
    for drop_row in range(rows):
        can_place = True
        
        # Проверяем возможность размещения
        for r, c in piece:
            new_r, new_c = drop_row + r, col_offset + c
            
            # Проверяем выход за границы или столкновение с другими блоками
            if (new_r >= rows or new_c < 0 or new_c >= cols or 
                (new_r >= 0 and new_board[new_r][new_c] == 1)):
                can_place = False
                break
        
        if not can_place and drop_row == 0:
            # Невозможно разместить фигуру
            return None
        
        if not can_place:
            # Размещаем фигуру на позицию выше
            drop_row -= 1
            for r, c in piece:
                new_r, new_c = drop_row + r, col_offset + c
                if 0 <= new_r < rows and 0 <= new_c < cols:
                    new_board[new_r][new_c] = 2  # Помечаем как активную фигуру
            return new_board
    
    # Если дошли до конца поля без столкновений
    for r, c in piece:
        new_r, new_c = rows - 1, col_offset + c
        if 0 <= new_r < rows and 0 <= new_c < cols:
            new_board[new_r][new_c] = 2
    
    return new_board

def count_holes(board):
    """Подсчитывает количество дыр в поле (пустые ячейки с блоком выше)"""
    holes = 0
    
    for c in range(cols):
        block_found = False
        for r in range(rows):
            if board[r][c] in [1, 2]:  # Блок
                block_found = True
            elif block_found and board[r][c] == 0:  # Дыра
                holes += 1
    
    return holes

def calculate_tower_heights(board):
    """Подсчитывает высоту башен для каждой колонки"""
    heights = []
    
    for c in range(cols):
        height = 0
        for r in range(rows):
            if board[r][c] in [1, 2]:
                height = rows - r
                break
        heights.append(height)
    
    return heights

def count_high_towers(heights, threshold=3):
    """Подсчитывает количество высоких башен"""
    high_towers = 0
    
    for h in heights:
        if h > threshold:
            high_towers += h - threshold  # Добавляем штраф за каждый лишний блок
    
    return high_towers

def calculate_bumpiness(heights):
    """Подсчитывает неровность поверхности"""
    bumpiness = 0
    
    for i in range(1, len(heights)):
        bumpiness += abs(heights[i] - heights[i-1])
    
    return bumpiness

def calculate_row_transitions(board):
    """Подсчитывает переходы между блоками и пустыми клетками в строках"""
    transitions = 0
    
    for r in range(rows):
        row_transitions = 0
        prev_cell = 1  # Считаем, что слева от доски стена
        
        for c in range(cols):
            current_cell = 1 if board[r][c] in [1, 2] else 0
            if current_cell != prev_cell:
                row_transitions += 1
            prev_cell = current_cell
        
        # Добавляем переход в правой стене
        if prev_cell == 0:
            row_transitions += 1
        
        transitions += row_transitions
    
    return transitions

def calculate_column_transitions(board):
    """Подсчитывает переходы между блоками и пустыми клетками в столбцах"""
    transitions = 0
    
    for c in range(cols):
        col_transitions = 0
        prev_cell = 1  # Считаем, что снизу доски стена
        
        for r in range(rows-1, -1, -1):
            current_cell = 1 if board[r][c] in [1, 2] else 0
            if current_cell != prev_cell:
                col_transitions += 1
            prev_cell = current_cell
        
        transitions += col_transitions
    
    return transitions

def count_completed_lines(board):
    """Подсчитывает количество заполненных строк"""
    completed = 0
    
    for r in range(rows):
        if all(board[r][c] in [1, 2] for c in range(cols)):
            completed += 1
    
    return completed

def evaluate_board(board):
    """Оценивает качество доски по нескольким критериям"""
    # Подсчитываем различные метрики
    holes = count_holes(board)
    heights = calculate_tower_heights(board)
    high_towers = count_high_towers(heights)
    bumpiness = calculate_bumpiness(heights)
    row_transitions = calculate_row_transitions(board)
    col_transitions = calculate_column_transitions(board)
    completed = count_completed_lines(board)
    aggregate_height = sum(heights)
    
    # Веса для различных мет рик (настройте по результатам тестирования)
    weights = {
        'completed_lines': 1.0,    # Поощрение за собранные линии (чем больше, тем лучше)
        'holes': -100.0,            # Штраф за дыры (пустые клетки под блоками)
        'high_towers': -1,        # Штраф за высоки е столбцы
        'bumpiness': -0.5,          # Штраф за неровную поверхность
        'aggregate_height': -1,   # Штраф за общую высоту поля
        'row_transitions': -0.5,      # Штраф за перепады в строках
        'col_transitions': -1       # Штраф за перепады в столбцах
    }
     
    # Вычисляем итоговую оценку
    score = (
        weights['completed_lines'] * completed +    
        weights['holes'] * holes +
        weights['high_towers'] * high_towers +
        weights['bumpiness'] * bumpiness +
        weights['aggregate_height'] * aggregate_height +
        weights['row_transitions'] * row_transitions +
        weights['col_transitions'] * col_transitions
    )
    
    return score

def find_best_move(board, piece_shape):
    """Находит лучший ход для текущей фигуры"""
    if not piece_shape:
        return None, None, -float('inf')
    
    # Получаем все возможные повороты
    rotations = get_all_possible_rotations(piece_shape)
    
    best_score = -float('inf')
    best_rotation = None
    best_column = None
    
    # Перебираем все возможные повороты и позиции
    for rotation in rotations:
        # Определяем ширину фигуры
        max_c = max(c for _, c in rotation) if rotation else 0
        
        # Перебираем все возможные позиции по горизонтали
        for col_offset in range(-min(c for _, c in rotation), cols - max_c):
            # Симулируем падение
            result_board = simulate_drop(board, rotation, col_offset)
            
            if result_board:
                # Оцениваем результат
                score = evaluate_board(result_board)
                
                if score > best_score:
                    best_score = score
                    best_rotation = rotation
                    best_column = col_offset
    
    return best_rotation, best_column, best_score

def calculate_final_shadow_position(board, piece_shape, col_offset):
    """Рассчитывает конечную позицию тени"""
    if not piece_shape:
        return None
    
    # Находим конечную позицию падения
    final_row = 0
    for drop_row in range(rows):
        can_place = True
        
        # Проверяем возможность размещения
        for r, c in piece_shape:
            new_r, new_c = drop_row + r, col_offset + c
            
            # Проверяем выход за границы или столкновение с другими блоками
            if (new_r >= rows or new_c < 0 or new_c >= cols or 
                (new_r >= 0 and board[new_r][new_c] == 1)):
                can_place = False
                break
        
        if not can_place:
            break
        final_row = drop_row
    
    # Создаем позиции тени
    shadow_positions = []
    for r, c in piece_shape:
        shadow_positions.append((final_row + r, col_offset + c))
    
    return shadow_positions

def shadow_matches_target(current_shadow, target_shadow):
    """Проверяет, совпадает ли текущая тень с целевой"""
    if not current_shadow or not target_shadow:
        return False
    
    # Сортируем позиции для лучшего сравнения
    current_set = set(current_shadow[0])
    target_set = set(target_shadow)
    
    # Проверяем, совпадают ли наборы позиций
    return current_set == target_set

def keyboard_listener():
    """Слушает клавиатуру для управления AI"""
    global ai_enabled, stop_flag
    
    while not stop_flag:
        if keyboard.is_pressed('q'):
            print ("Остановка работы...")
            stop_flag = True
        
        if keyboard.is_pressed('a'):
            ai_enabled = not ai_enabled
            print(f"AI {'включен' if ai_enabled else 'выключен'}")
            time.sleep(0.3)  # Предотвращение многократного переключения
        
       

def print_matrix(matrix):
    """Красиво выводит матрицу"""
    symbols = {0: "□", 1: "■", 2: "▣", 3: "░"}  # Добавлен символ для тени
    for row in matrix:
        print(" ".join(symbols[cell] for cell in row))
    print("-" * 20)

def execute_move_using_shadow():
    """Выполняет оптимальный ход, основываясь на тени"""
    # Получаем текущее состояние игры
    current = get_game_matrix()
    processed = identify_active_piece_with_shadow(current)
    
    # Извлекаем тень и статичную доску
    shadow_info = extract_shadow_position(current)
    current_board = get_static_board(processed)
    
    if not shadow_info:
        print("Не удалось определить тень")
        return
    
    # Определяем тип тетромино на основе тени
    shadow_cells, shadow_row, shadow_col = shadow_info
    
    # Определяем полную форму тетромино на основе тени
    piece_shape = identify_tetromino_from_shadow(shadow_info)
    
    if not piece_shape:
        print("Не удалось определить форму тетромино по тени")
        return
    
    print(f"Определена форма фигуры по тени (блоков: {len(piece_shape)})")
    
    # Находим лучший ход
    best_rotation, best_column, best_score = find_best_move(current_board, piece_shape)
    
    if best_rotation is None:
        print("Не удалось найти оптимальный ход")
        return
    
    print(f"Лучший ход: колонка {best_column} (оценка: {best_score:.2f})")
    
    # Рассчитываем целевую позицию тени
    target_shadow = calculate_final_shadow_position(current_board, best_rotation, best_column)
    
    # Выполняем необходимые повороты
    all_rotations = get_all_possible_rotations(piece_shape)
    
    # Определяем текущий поворот и целевой поворот
    current_rotation = None
    for rot in all_rotations:
        target_shadow_test = calculate_final_shadow_position(current_board, rot, shadow_col)
        if target_shadow_test and shadow_matches_target(shadow_info, target_shadow_test):
            current_rotation = rot
            break
    
    if not current_rotation:
        current_rotation = piece_shape  # Используем определенную форму как текущую
    
    current_rot_index = next((i for i, rot in enumerate(all_rotations) if rot == current_rotation), 0)
    target_rot_index = next((i for i, rot in enumerate(all_rotations) if rot == best_rotation), 0)
    rotations_needed = (target_rot_index - current_rot_index) % len(all_rotations)
    
    print(f"Необходимо сделать {rotations_needed} поворотов")
    
    # Выполняем повороты
    for _ in range(rotations_needed):
        keyboard.press_and_release('up')
        time.sleep(0.01)    
    
    # Обновляем состояние после поворотов
    current = get_game_matrix()
    shadow_info = extract_shadow_position(current)
    
    if not shadow_info:
        print("Не удалось определить тень после поворотов")
        return
    
    # Определяем направление движения для достижения целевой позиции
    _, _, current_shadow_col = shadow_info
    
    # Двигаем фигуру горизонтально, пока тень не совпадет с целевой
    moves_count = 0
    max_moves = 20  # Ограничение на количество перемещений
    
    print(f"Текущая позиция тени: {current_shadow_col}, целевая: {best_column}")
    
    while current_shadow_col != best_column and moves_count < max_moves:
        if current_shadow_col < best_column:
            keyboard.press_and_release('right')
            print("Движение вправо")
        else:
            keyboard.press_and_release('left')
            print("Движение влево")
        
        moves_count += 1
        time.sleep(0.01)
        
        # Обновляем позицию тени
        current = get_game_matrix()
        shadow_info = extract_shadow_position(current)
        
        if not shadow_info:
            print("Потеряна тень при горизонтальном перемещении")
            break
        
        _, _, current_shadow_col = shadow_info
    
    # Финальная проверка перед хард-дропом
    current = get_game_matrix()
    shadow_info = extract_shadow_position(current)
    
    if shadow_info:
        _, _, final_shadow_col = shadow_info
        print(f"Финальная позиция тени: {final_shadow_col}, целевая: {best_column}")
        
        if final_shadow_col == best_column:
            print("Тень в нужной позиции, выполняю хард-дроп")
            keyboard.press_and_release('space')
        else:
            print("Тень не в нужной позиции, отменяем хард-дроп")
    else:
        print("Тень не обнаружена, отменяем хард-дроп")

def main():
    global piece_detected_frames, stop_flag
    
    prev_matrix = None
    prev_shadow = None
    prev_board = None
    
    print("Запуск оптимизированного Tetris AI с использованием тени...")
    print("□ - пусто, ■ - статичный блок, ▣ - активный блок, ░ - тень")
    print("Управление: q - выход, a - включить/выключить AI")
    print("-" * 40)
    
    # Запускаем поток для отслеживания клавиатуры
    keyboard_thread = threading.Thread(target=keyboard_listener)
    keyboard_thread.daemon = True
    keyboard_thread.start()
    
    try:
        while not stop_flag:
            current = get_game_matrix()
            processed = identify_active_piece_with_shadow(current)
            
            # Извлекаем тень и статичную доску
            shadow_info = extract_shadow_position(current)
            current_board = get_static_board(processed)
            
            if shadow_info:
                shadow_cells, shadow_row, shadow_col = shadow_info
                
                # Проверяем, является ли это новой тенью
                if prev_shadow != shadow_cells or prev_board != current_board:
                    piece_detected_frames = 0
                    prev_shadow = shadow_cells
                    prev_board = current_board
                else:
                    piece_detected_frames += 1
                
                # Если тень стабильно определяется и AI включен
                if ai_enabled and piece_detected_frames >= MIN_FRAMES_TO_START:
                    print("Запуск расчета с использованием тени...")
                    execute_move_using_shadow()

                    # Сбрасываем счетчик кадров
                    piece_detected_frames = 0
            
            if prev_matrix is None or not np.array_equal(prev_matrix, processed):
                print("Обновление поля:")
                print_matrix(processed)
                prev_matrix = processed
            
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nОстановлено пользователем")
    except Exception as e:
        print(f"Ошибка: {e}")
    finally:
        stop_flag = True
        print("Завершение работы...")

if __name__ == "__main__":
    main()