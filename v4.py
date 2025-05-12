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
    
    # Веса для различных метрик (настройте по результатам тестирования)
    weights = {
        'completed_lines': 10.0,
        'holes': -100.0 ,
        'high_towers': 0,
        'bumpiness': -1,
        'aggregate_height': -1,
        'row_transitions': -1,
        'col_transitions': -1
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

def keyboard_listener():
    """Слушает клавиатуру для управления AI"""
    global ai_enabled, stop_flag
    
    while not stop_flag:
        if keyboard.is_pressed('q'):
            print("Остановка работы...")
            stop_flag = True
        
        if keyboard.is_pressed('a'):
            ai_enabled = not ai_enabled
            print(f"AI {'включен' if ai_enabled else 'выключен'}")
            time.sleep(0.3)  # Предотвращение многократного переключения
        
        time.sleep(0.1)

def print_matrix(matrix):
    """Красиво выводит матрицу"""
    symbols = {0: "□", 1: "■", 2: "▣", 3: "░"}  # Добавлен символ для тени
    for row in matrix:
        print(" ".join(symbols[cell] for cell in row))
    print("-" * 20)

def execute_moves_with_recalculation():
    """Выполняет оптимальный ход с пересчетом после каждого действия"""
    # Получаем текущее состояние игры
    current = get_game_matrix()
    processed = identify_active_piece_with_shadow(current)
    
    # Извлекаем активную фигуру и статичную доску
    piece_info = extract_piece_shape(processed)
    
    if not piece_info:
        print("Не удалось определить активную фигуру")
        return
    
    piece_shape, piece_row, piece_col = piece_info
    current_board = get_static_board(processed)
    
    # Сначала спускаем фигуру, пока не увидим все 4 блока
    while len(piece_shape) < 4 :
        #keyboard.press_and_release('up')
        #time.sleep(0.01)
        
        # Обновляем состояние после спуска
        current = get_game_matrix()
        processed = identify_active_piece_with_shadow(current)
        piece_info = extract_piece_shape(processed)
        
        if not piece_info:
            print("Потеряна фигура при спуске")
            return
        
        piece_shape, piece_row, piece_col = piece_info
    
    print(f"Фигура полностью видна (блоков: {len(piece_shape)})")
    
    # Теперь выполняем обычную логику поиска лучшего хода
    best_rotation, best_column, best_score = find_best_move(current_board, piece_shape)
    
    if best_rotation is None:
        print("Не удалось найти оптимальный ход")
        return
    
    print(f"Начальный расчет: лучший ход (оценка: {best_score:.2f})")
    
    # Остальная часть функции остается без изменений
    all_rotations = get_all_possible_rotations(piece_shape)
    current_rot_index = next((i for i, rot in enumerate(all_rotations) if rot == piece_shape), 0)
    target_rot_index = next((i for i, rot in enumerate(all_rotations) if rot == best_rotation), 0)
    rotations_needed = (target_rot_index - current_rot_index) % len(all_rotations)
    
    for _ in range(rotations_needed):
        keyboard.press_and_release('up')
        time.sleep(0.01)
        
        new_state = get_game_matrix()
        new_processed = identify_active_piece_with_shadow(new_state)
        new_piece_info = extract_piece_shape(new_processed)
        
        if not new_piece_info:
            print("Потеряна фигура после поворота")
            continue
        
        new_piece_shape, new_piece_row, new_piece_col = new_piece_info
        new_board = get_static_board(new_processed)
        
        new_best_rotation, new_best_column, new_score = find_best_move(new_board, new_piece_shape)
        
        if new_best_rotation is not None:
            best_rotation = new_best_rotation
            best_column = new_best_column
            best_score = new_score
            print(f"После поворота: новый лучший ход (оценка: {best_score:.2f})")
    
    current = get_game_matrix()
    processed = identify_active_piece_with_shadow(current)
    piece_info = extract_piece_shape(processed)
    
    if not piece_info:
        print("Не удалось определить активную фигуру после поворотов")
        return
    
    _, _, piece_col = piece_info
    
    while piece_col != best_column:
        if piece_col < best_column:
            keyboard.press_and_release('right')
            piece_col += 1
        else:
            keyboard.press_and_release('left')
            piece_col -= 1
        
        time.sleep(0.01)
        
        new_state = get_game_matrix()
        new_processed = identify_active_piece_with_shadow(new_state)
        new_piece_info = extract_piece_shape(new_processed)
        
        if not new_piece_info:
            print("Потеряна фигура после горизонтального перемещения")
            break
        
        new_piece_shape, new_piece_row, new_piece_col = new_piece_info
        new_board = get_static_board(new_processed)
        
        piece_col = new_piece_col
        
        new_best_rotation, new_best_column, new_score = find_best_move(new_board, new_piece_shape)
        
        if new_best_rotation is not None:
            best_rotation = new_best_rotation
            best_column = new_best_column
            best_score = new_score
            print(f"После перемещения: новый лучший ход (оценка: {best_score:.2f})")
    
    print("Выполняю хард-дроп")
    keyboard.press_and_release('space')
    #time.sleep(0.2)

def main():
    global piece_detected_frames, stop_flag
    
    prev_matrix = None
    prev_piece = None
    prev_board = None
    
    print("Запуск анализатора Tetris AI с определением тени...")
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
            
            # Извлекаем активную фигуру и статичную доску
            piece_info = extract_piece_shape(processed)
            
            if piece_info:
                piece_shape, piece_row, piece_col = piece_info
                current_board = get_static_board(processed)
                
                # Проверяем, является ли это новой фигурой
                if prev_piece != piece_shape or prev_board != current_board:
                    piece_detected_frames = 0
                    prev_piece = piece_shape
                    prev_board = current_board
                else:
                    piece_detected_frames += 1
                
                # Если фигура стабильно определяется и AI включен
                if ai_enabled and piece_detected_frames >= MIN_FRAMES_TO_START:
                    print("Запуск расчета с пересчетом после каждого действия...")
                    execute_moves_with_recalculation()

                    # Сбрасываем счетчик кадров
                    piece_detected_frames = 0
            
            if prev_matrix is None or not np.array_equal(prev_matrix, processed):
                print("Обновление поля:")
                print_matrix(processed)
                prev_matrix = processed
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nОстановлено пользователем")
    except Exception as e:
        print(f"Ошибка: {e}")
    finally:
        stop_flag = True
        print("Завершение работы...")

if __name__ == "__main__":
    main()