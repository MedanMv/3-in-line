import cv2
from PIL import ImageGrab
import numpy as np
from pynput.mouse import Button, Controller             #FINALLL make swapping when not all cells seen
import time

mouse = Controller()

roi = (225, 129, 952, 857)  # depends on the screen
kernelMask = np.ones((3, 3), np.uint8)

cell_centers = [
    [(217, 141), (289, 141), (358, 141), (434, 141), (505, 141), (578, 141), (652, 141), (725, 141)],
    [(217, 214), (289, 214), (358, 214), (434, 214), (505, 214), (578, 214), (652, 214), (725, 214)],
    [(217, 283), (289, 283), (358, 283), (434, 283), (505, 283), (578, 283), (652, 283), (725, 283)],
    [(217, 357), (289, 357), (358, 357), (434, 357), (505, 357), (578, 357), (652, 357), (725, 357)],
    [(217, 433), (289, 433), (358, 433), (434, 433), (505, 433), (578, 433), (652, 433), (725, 433)], # depends on the screen
    [(217, 504), (289, 504), (358, 504), (434, 504), (505, 504), (578, 504), (652, 504), (725, 504)],
    [(217, 574), (289, 574), (358, 574), (434, 574), (505, 574), (578, 574), (652, 574), (725, 574)],
    [(217, 646), (289, 646), (358, 646), (434, 646), (505, 646), (578, 646), (652, 646), (725, 646)],
]

field_array = [[0] * 8 for _ in range(8)]
color_ranges = [
    ((123, 99, 71), (156, 127, 111)), # cm
    ((97, 43, 106), (121, 72, 134)),
    ((50, 95, 100), (79, 130, 151)), # wywern
    ((43, 53, 102), (89, 112, 167)),  # lich
    ((45, 21, 29), (103, 89, 140)),  # brood
    ((106, 38, 33), (170, 96, 69))   # lina
]

def combine_cells(cells, cell_width, cell_height, num_rows, num_cols):
    combined_image = np.zeros((cell_height * num_rows, cell_width * num_cols, 3), dtype=np.uint8)
    index = 0
    for row in range(num_rows):
        for col in range(num_cols):
            x1 = col * cell_width
            y1 = row * cell_height
            x2 = x1 + cell_width
            y2 = y1 + cell_height
            combined_image[y1:y2, x1:x2] = cells[index]
            index += 1
    return combined_image

def capture_screen(roi=None):
    screen = ImageGrab.grab()
    screen_np = np.array(screen)
    if roi:
        x1, y1, x2, y2 = roi
        screen_bgr1 = screen_np[y1:y2, x1:x2]
        eroded_mask = cv2.erode(screen_bgr1, kernelMask, iterations=2)
        dilated_mask = cv2.dilate(eroded_mask, kernelMask, iterations=2)
        opened = cv2.morphologyEx(dilated_mask, cv2.MORPH_OPEN, kernelMask)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernelMask)
        opened1 = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernelMask)
        closed1 = cv2.morphologyEx(opened1, cv2.MORPH_CLOSE, kernelMask)
        bluur = cv2.GaussianBlur(closed1, (5, 5), sigmaX=1)
    return bluur

def split_into_cells(image, grid_size=(8, 8)):
    height, width, _ = image.shape
    num_rows, num_cols = grid_size
    cell_width = width // num_cols
    cell_height = height // num_rows
    cells = []
    for row in range(num_rows):
        for col in range(num_cols):
            x1 = col * cell_width
            y1 = row * cell_height
            x2 = (col + 1) * cell_width
            y2 = (row + 1) * cell_height
            cell = image[y1:y2, x1:x2]
            cells.append(cell)
    return cells, cell_width, cell_height, num_rows, num_cols

def process_cell(cell, color_ranges, row, col):
    # Apply Gaussian blur to smooth the image
    processed_cell = cv2.GaussianBlur(cell, (5, 5), sigmaX=1)

    # Calculate the average color in BGR format
    average_color = cv2.mean(processed_cell)[:3]  # Extract BGR channels
    average_color = tuple(int(round(c)) for c in average_color)  # Convert color values to integers

    # Iterate through the defined color ranges to find a match
    for idx, (lower_bound, upper_bound) in enumerate(color_ranges):
        # Check if the average color falls within the current color range
        if all(lower <= val <= upper for lower, val, upper in zip(lower_bound, average_color, upper_bound)):
            return idx + 1, None  # If a match is found, return the color index and no undetected color

    # If no color match is found, return 0 and the average color for debugging
    return 0, average_color

def process_frame(frame):
    cells, cell_width, cell_height, num_rows, num_cols = split_into_cells(frame)
    all_cells_detected = True
    for i in range(num_rows):
        for j in range(num_cols):
            result, undetected_color = process_cell(cells[i * num_cols + j], color_ranges, i, j)
            field_array[i][j] = result

            # If the cell is undetected, print its position and average color
            if result == 0:
                all_cells_detected = False
                print(f"Undetected cell at ({i}, {j}) with average color: {undetected_color}")

    combined_image = combine_cells(cells, cell_width, cell_height, num_rows, num_cols)

    if all_cells_detected:
        time.sleep(0.5)
        # new_frame = capture_screen(roi)
        print("No changes detected, searching for the best swap...")
        check_and_mark_best_swap(field_array)

    return combined_image

def check_and_mark_best_swap(field_array):
    rows = len(field_array)
    cols = len(field_array[0])
    best_swap = None
    max_sequence_length = 0

    # x1, y1, x2, y2 = roi
    # width = x2 - x1
    # height = y2 - y1
    # num_rows, num_cols = grid_size
    # cell_width = width // num_cols
    # cell_height = height // num_rows

    def count_sequence_length(arr):
        max_length = 1
        current_length = 1
        for i in range(1, len(arr)):
            if arr[i] == arr[i - 1] and arr[i] != 0:
                current_length += 1
            else:
                max_length = max(max_length, current_length)
                current_length = 1
        max_length = max(max_length, current_length)
        return max_length

    def simulate_swap_and_evaluate(r1, c1, r2, c2):
        simulated_field = [row[:] for row in field_array]
        simulated_field[r1][c1], simulated_field[r2][c2] = simulated_field[r2][c2], simulated_field[r1][c1]
        row1_len = count_sequence_length(simulated_field[r1])
        row2_len = count_sequence_length(simulated_field[r2])
        col1_len = count_sequence_length([simulated_field[r][c1] for r in range(rows)])
        col2_len = count_sequence_length([simulated_field[r][c2] for r in range(rows)])
        return max(row1_len, row2_len, col1_len, col2_len)

    for r in range(rows):
        for c in range(cols):
            if c < cols - 1:
                seq_length = simulate_swap_and_evaluate(r, c, r, c + 1)
                if seq_length > max_sequence_length:
                    max_sequence_length = seq_length
                    best_swap = (r, c, r, c + 1)
            if r < rows - 1:
                seq_length = simulate_swap_and_evaluate(r, c, r + 1, c)
                if seq_length > max_sequence_length:
                    max_sequence_length = seq_length
                    best_swap = (r, c, r + 1, c)

    if best_swap:
        r1, c1, r2, c2 = best_swap
        print(f"Best swap between cell[{r1}, {c1}] at center {cell_centers[r1][c1]} and cell[{r2}, {c2}] at center {cell_centers[r2][c2]}")
        run_function_once(r1, c1, r2, c2)

def run_function_once(r1, c1, r2, c2):
    center1 = cell_centers[r1][c1]
    center2 = cell_centers[r2][c2]
    
    print(f"Running function for swap between ({r1},{c1}) at center {center1} and ({r2},{c2}) at center {center2}")

    # Move the mouse to the first position and give some time before pressing
    mouse.position = center1
    time.sleep(0.1)  # Add a slight delay before pressing the button
    mouse.press(Button.left)
    time.sleep(0.1)  # Increase delay after pressing

    num_steps = 33
    step_y = int((center2[1] - center1[1]) / num_steps)
    step_x = int((center2[0] - center1[0]) / num_steps)

    for step in range(num_steps):
        mouse.move(0, step_y)
        time.sleep(0.003)
        mouse.move(step_x, 0)
        time.sleep(0.003)

    mouse.release(Button.left)
    time.sleep(0.1)  # Add a delay after releasing the button

    field_array[r1][c1] = 0
    field_array[r2][c2] = 0
    
    mouse.position = (0, 0)
    time.sleep(0.3) 

    print("Updated field_array after swap:")
    for row in field_array:
        print(row)

while True:
    frame = capture_screen(roi)
    processed_frame = process_frame(frame)
    cv2.imshow('Processed ROI Capture', processed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()