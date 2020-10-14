import numpy as np

def convolution(input, kernel, stride=1, padding=0):
    # Apply padding to input
    input = np.pad(input, padding, 'constant')
    # Unpack kernel and input shape
    (kernel_rows, kernel_cols) = kernel.shape
    (input_rows, input_cols) = input.shape

    # Convolution array dimensions
    conv_rows = int(np.floor((input_rows - kernel_rows)/stride)) + 1
    conv_cols = int(np.floor((input_cols - kernel_cols)/stride)) + 1
    # Create 0-filled convolution array
    conv = np.zeros((conv_rows, conv_rows))

    # Iterate over convolution array
    for row in range(conv_rows):
        for col in range(conv_cols):
            # Input slice start position
            row_input = row * stride
            col_input = col * stride
            # Slice input with dimensions of kernel
            input_slice = np.array(input[
                row_input : row_input + kernel_rows,
                col_input : col_input + kernel_cols
            ])
            # Fill conv array with element-wise multiplication then sum with einstein summation
            conv[row, col] =  np.einsum('ij,ij', input_slice, kernel)

    return conv


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

def flatten(input):
    return np.ravel(input)

test = np.array([[1, 2, 3], [4, 5, 6]])
print(flatten(test))