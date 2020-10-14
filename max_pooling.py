import numpy as np

def max_pool(input, kernel_size, stride=1):
    # Unpack input array dimensions
    (input_rows, input_cols) = input.shape

    # Output array dimensions
    output_rows = int(np.floor((input_rows - kernel_size)/stride)) + 1
    output_cols = int(np.floor((input_cols - kernel_size)/stride)) + 1
    # Create 0-filled output array
    output = np.zeros((output_rows, output_cols))

    # Iterate over output array
    for row in range(output_rows):
        for col in range(output_cols):
            # Input slice start position
            row_input = row * stride
            col_input = col * stride
            # Slice input with kernel dimensions
            input_slice = np.array(input[
                row_input : row_input + kernel_size,
                col_input : col_input + kernel_size
            ])
            # Fill output array with max value
            output[row, col] = np.amax(input_slice)

    return output