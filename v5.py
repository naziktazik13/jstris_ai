from PIL import ImageGrab
import numpy as np
import time
from collections import deque
import keyboard
import threading

x, y, width, height = 580, 213, 250, 500
cols, rows = 10, 20
cell_width = width // cols
cell_height = height // rows

WHITE = (255, 255, 255)
GRAY = (106, 106, 106)
BGRAY = (153, 153, 153)
SHADOW = (127, 127, 127)
BACKGROUND = (0, 0, 0)

ai_enabled = True
stop_flag = False 
piece_detected_frames = 1
MIN_FRAMES_TO_START = 1 

TETROMINO_SHAPES = {
    'I': [(0, 0), (0, 1), (0, 2), (0, 3)],
    'O': [(0, 0), (0, 1), (1, 0), (1, 1)],
    'T': [(0, 1), (1, 0), (1, 1), (1, 2)],
    'S': [(0, 1), (0, 2), (1, 0), (1, 1)],
    'Z': [(0, 0), (0, 1), (1, 1), (1, 2)],
    'J': [(0, 0), (1, 0), (1, 1), (1, 2)],
    'L': [(0, 2), (1, 0), (1, 1), (1, 2)]
}

def get_game_matrix():
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
                matrix_row.append(1)
            elif pixel == GRAY or pixel == BGRAY:
                matrix_row.append(2)
            elif pixel == SHADOW:
                matrix_row.append(3)
            else:
                matrix_row.append(0)
        matrix.append(matrix_row)
    return matrix

def identify_active_piece_with_shadow(matrix):
    result = [[0 for _ in range(cols)] for _ in range(rows)]
    
    shadow_positions = []
    for r in range(rows):
        for c in range(cols):
            if matrix[r][c] == 3:
                shadow_positions.append((r, c))
    
    if not shadow_positions:
        for r in range(rows):
            for c in range(cols):
                if matrix[r][c] == 2:
                    result[r][c] = 1
                elif matrix[r][c] == 1:
                    result[r][c] = 2
        return result
    
    shadow_columns = set(c for _, c in shadow_positions)
    
    shadow_highest_row = {}
    for r, c in shadow_positions:
        if c not in shadow_highest_row or r < shadow_highest_row[c]:
            shadow_highest_row[c] = r
    
    for r in range(rows):
        for c in range(cols):
            if matrix[r][c] == 0:
                continue
                
            if matrix[r][c] == 3:
                continue
                
            if matrix[r][c] == 2:
                result[r][c] = 1
                continue
                
            if c in shadow_columns and r < shadow_highest_row[c]:
                result[r][c] = 2
            else:
                result[r][c] = 1
    
    return result

def extract_piece_shape(matrix):
    piece_cells = []
    
    for r in range(rows):
        for c in range(cols):
            if matrix[r][c] == 2:
                piece_cells.append((r, c))
    
    if not piece_cells:
        return None
    
    min_r = min(r for r, c in piece_cells)
    min_c = min(c for r, c in piece_cells)
    
    normalized_piece = [(r - min_r, c - min_c) for r, c in piece_cells]
    return normalized_piece, min_r, min_c

def extract_shadow_position(matrix):
    shadow_cells = []
    
    for r in range(rows):
        for c in range(cols):
            if matrix[r][c] == 3:
                shadow_cells.append((r, c))
    
    if not shadow_cells:
        return None
    
    min_r = min(r for r, c in shadow_cells)
    min_c = min(c for r, c in shadow_cells)
    
    normalized_shadow = [(r - min_r, c - min_c) for r, c in shadow_cells]
    return normalized_shadow, min_r, min_c

def identify_tetromino_from_shadow(shadow_cells):
    if not shadow_cells:
        return None
    
    normalized_shadow = set(shadow_cells[0])
    
    for shape_name, shape in TETROMINO_SHAPES.items():
        rotations = get_all_possible_rotations(shape)
        for rotation in rotations:
            rotation_set = set(rotation)
            if len(rotation_set) == len(normalized_shadow) and all(cell in rotation_set for cell in normalized_shadow):
                return rotation
    
    return shadow_cells[0]

def get_static_board(matrix):
    board = [[0 for _ in range(cols)] for _ in range(rows)]
    
    for r in range(rows):
        for c in range(cols):
            if matrix[r][c] == 1:
                board[r][c] = 1
    
    return board

def get_all_possible_rotations(piece_shape):
    if not piece_shape:
        return []
    
    rotations = []
    current_rotation = piece_shape
    
    for _ in range(4):
        rotations.append(current_rotation)
        current_rotation = [(c, -r) for r, c in current_rotation]
        min_r = min(r for r, _ in current_rotation)
        min_c = min(c for _, c in current_rotation)
        current_rotation = [(r - min_r, c - min_c) for r, c in current_rotation]
        
        if current_rotation in rotations:
            break
    
    return rotations

def simulate_drop(board, piece, col_offset):
    if not piece:
        return None
    
    new_board = [row[:] for row in board]
    
    for drop_row in range(rows):
        can_place = True
        
        for r, c in piece:
            new_r, new_c = drop_row + r, col_offset + c
            
            if (new_r >= rows or new_c < 0 or new_c >= cols or 
                (new_r >= 0 and new_board[new_r][new_c] == 1)):
                can_place = False
                break
        
        if not can_place and drop_row == 0:
            return None
        
        if not can_place:
            drop_row -= 1
            for r, c in piece:
                new_r, new_c = drop_row + r, col_offset + c
                if 0 <= new_r < rows and 0 <= new_c < cols:
                    new_board[new_r][new_c] = 2
            return new_board
    
    for r, c in piece:
        new_r, new_c = rows - 1, col_offset + c
        if 0 <= new_r < rows and 0 <= new_c < cols:
            new_board[new_r][new_c] = 2
    
    return new_board

def count_holes(board):
    holes = 0
    
    for c in range(cols):
        block_found = False
        for r in range(rows):
            if board[r][c] in [1, 2]:
                block_found = True
            elif block_found and board[r][c] == 0:
                holes += 1
    
    return holes

def calculate_tower_heights(board):
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
    high_towers = 0
    
    for h in heights:
        if h > threshold:
            high_towers += h - threshold
    
    return high_towers

def calculate_bumpiness(heights):
    bumpiness = 0
    
    for i in range(1, len(heights)):
        bumpiness += abs(heights[i] - heights[i-1])
    
    return bumpiness

def calculate_row_transitions(board):
    transitions = 0
    
    for r in range(rows):
        row_transitions = 0
        prev_cell = 1
        
        for c in range(cols):
            current_cell = 1 if board[r][c] in [1, 2] else 0
            if current_cell != prev_cell:
                row_transitions += 1
            prev_cell = current_cell
        
        if prev_cell == 0:
            row_transitions += 1
        
        transitions += row_transitions
    
    return transitions

def calculate_column_transitions(board):
    transitions = 0
    
    for c in range(cols):
        col_transitions = 0
        prev_cell = 1
        
        for r in range(rows-1, -1, -1):
            current_cell = 1 if board[r][c] in [1, 2] else 0
            if current_cell != prev_cell:
                col_transitions += 1
            prev_cell = current_cell
        
        transitions += col_transitions
    
    return transitions

def count_completed_lines(board):
    completed = 0
    
    for r in range(rows):
        if all(board[r][c] in [1, 2] for c in range(cols)):
            completed += 1
    
    return completed

def evaluate_board(board):
    holes = count_holes(board)
    heights = calculate_tower_heights(board)
    high_towers = count_high_towers(heights)
    bumpiness = calculate_bumpiness(heights)
    row_transitions = calculate_row_transitions(board)
    col_transitions = calculate_column_transitions(board)
    completed = count_completed_lines(board)
    aggregate_height = sum(heights)
    
    weights = {
        'completed_lines': 10.0,
        'holes': -5.0,
        'high_towers': -3,
        'bumpiness': -1,
        'aggregate_height': -2,
        'row_transitions': -0.5,
        'col_transitions': -5
    }
     
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
    if not piece_shape:
        return None, None, -float('inf')
    
    rotations = get_all_possible_rotations(piece_shape)
    
    best_score = -float('inf')
    best_rotation = None
    best_column = None
    
    for rotation in rotations:
        max_c = max(c for _, c in rotation) if rotation else 0
        
        for col_offset in range(-min(c for _, c in rotation), cols - max_c):
            result_board = simulate_drop(board, rotation, col_offset)
            
            if result_board:
                score = evaluate_board(result_board)
                
                if score > best_score:
                    best_score = score
                    best_rotation = rotation
                    best_column = col_offset
    
    return best_rotation, best_column, best_score

def calculate_final_shadow_position(board, piece_shape, col_offset):
    if not piece_shape:
        return None
    
    final_row = 0
    for drop_row in range(rows):
        can_place = True
        
        for r, c in piece_shape:
            new_r, new_c = drop_row + r, col_offset + c
            
            if (new_r >= rows or new_c < 0 or new_c >= cols or 
                (new_r >= 0 and board[new_r][new_c] == 1)):
                can_place = False
                break
        
        if not can_place:
            break
        final_row = drop_row
    
    shadow_positions = []
    for r, c in piece_shape:
        shadow_positions.append((final_row + r, col_offset + c))
    
    return shadow_positions

def shadow_matches_target(current_shadow, target_shadow):
    if not current_shadow or not target_shadow:
        return False
    
    current_set = set(current_shadow[0])
    target_set = set(target_shadow)
    
    return current_set == target_set

def keyboard_listener():
    global ai_enabled, stop_flag
    
    while not stop_flag:
        if keyboard.is_pressed('q'):
            print ("Остановка работы...")
            stop_flag = True
        
        if keyboard.is_pressed('a'):
            ai_enabled = not ai_enabled
            print(f"AI {'включен' if ai_enabled else 'выключен'}")
            time.sleep(0.3)
        
        time.sleep(0.1)

def print_matrix(matrix):
    symbols = {0: "□", 1: "■", 2: "▣", 3: "░"}
    for row in matrix:
        print(" ".join(symbols[cell] for cell in row))
    print("-" * 20)

def execute_move_using_shadow():
    current = get_game_matrix()
    processed = identify_active_piece_with_shadow(current)
    
    shadow_info = extract_shadow_position(current)
    current_board = get_static_board(processed)
    
    if not shadow_info:
        print("Не удалось определить тень")
        return
    
    shadow_cells, shadow_row, shadow_col = shadow_info
    
    piece_shape = identify_tetromino_from_shadow(shadow_info)
    
    if not piece_shape:
        print("Не удалось определить форму тетромино по тени")
        return
    
    print(f"Определена форма фигуры по тени (блоков: {len(piece_shape)})")
    
    best_rotation, best_column, best_score = find_best_move(current_board, piece_shape)
    
    if best_rotation is None:
        print("Не удалось найти оптимальный ход")
        return
    
    print(f"Лучший ход: колонка {best_column} (оценка: {best_score:.2f})")
    
    target_shadow = calculate_final_shadow_position(current_board, best_rotation, best_column)
    
    all_rotations = get_all_possible_rotations(piece_shape)
    
    current_rotation = None
    for rot in all_rotations:
        target_shadow_test = calculate_final_shadow_position(current_board, rot, shadow_col)
        if target_shadow_test and shadow_matches_target(shadow_info, target_shadow_test):
            current_rotation = rot
            break
    
    if not current_rotation:
        current_rotation = piece_shape
    
    current_rot_index = next((i for i, rot in enumerate(all_rotations) if rot == current_rotation), 0)
    target_rot_index = next((i for i, rot in enumerate(all_rotations) if rot == best_rotation), 0)
    rotations_needed = (target_rot_index - current_rot_index) % len(all_rotations)
    
    print(f"Необходимо сделать {rotations_needed} поворотов")
    
    for _ in range(rotations_needed):
        keyboard.press_and_release('up')
        time.sleep(0.01)    
    
    current = get_game_matrix()
    shadow_info = extract_shadow_position(current)
    
    if not shadow_info:
        print("Не удалось определить тень после поворотов")
        return
    
    _, _, current_shadow_col = shadow_info
    
    moves_count = 0
    max_moves = 20
    
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
        
        current = get_game_matrix()
        shadow_info = extract_shadow_position(current)
        
        if not shadow_info:
            print("Потеряна тень при горизонтальном перемещении")
            break
        
        _, _, current_shadow_col = shadow_info
    
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
    
    keyboard_thread = threading.Thread(target=keyboard_listener)
    keyboard_thread.daemon = True
    keyboard_thread.start()
    
    try:
        while not stop_flag:
            current = get_game_matrix()
            processed = identify_active_piece_with_shadow(current)
            
            shadow_info = extract_shadow_position(current)
            current_board = get_static_board(processed)
            
            if shadow_info:
                shadow_cells, shadow_row, shadow_col = shadow_info
                
                if prev_shadow != shadow_cells or prev_board != current_board:
                    piece_detected_frames = 0
                    prev_shadow = shadow_cells
                    prev_board = current_board
                else:
                    piece_detected_frames += 1
                
                if ai_enabled and piece_detected_frames >= MIN_FRAMES_TO_START:
                    print("Запуск расчета с использованием тени...")
                    execute_move_using_shadow()

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