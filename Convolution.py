import numpy as np

class Convolution:

    def __init__(self, n_filters, filter_shape, stride=1, padding=0):
        '''
            :param n_filters: number of filters
            :param filter_dim: filter dimensions tuple
            :param stride: stride value for convolution, default is 1
            :param padding: padding to apply to images, default is 0
        '''
        self.n_filters = n_filters
        (self.filter_rows, self.filter_cols) = filter_shape
        self.stride = stride
        self.padding = padding
        self.conv_filters = self.init_filters()

    def init_filters(self, scale = 1.0):
        '''
            Initializes filters.
            Normal distribution with std=1 and mean=0.
        '''
        # Create 0-filled array of filters with length self.n_filters
        filters = np.zeros((self.n_filters, self.filter_rows, self.filter_cols))
        # Iterate over filters
        for i in range(self.n_filters):
            # Replace filter i with normal distributed random values
            filters[i] = np.random.normal(loc = 0, scale = scale, size=filters[i].shape)
        return filters

    def conv_shape(self, input):
        '''
            Determines convolution operation output shape (tuple).
        '''
        # Unpack input shape
        (input_rows, input_cols) = input.shape
        # Determine convolution output shape and return
        conv_rows = int(np.floor((input_rows - self.filter_rows)/self.stride)) + 1
        conv_cols = int(np.floor((input_cols - self.filter_cols)/self.stride)) + 1
        return (conv_rows, conv_cols)

    def input_slice(self, input):
        '''
            Generator function, slices input into filter-sized arrays and yields them.
        '''
        # Unpack convolution output shape
        (conv_rows, conv_cols) = self.conv_shape(input)
        
        # Iterate over convolution array
        for row in range(conv_rows):
            for col in range(conv_cols):
                # Input slice start position
                row_input = row * self.stride
                col_input = col * self.stride
                # Slice input with dimensions of kernel
                input_slice = np.array(input[
                    row_input : row_input + self.filter_rows,
                    col_input : col_input + self.filter_cols
                ])
                yield input_slice, row, col

    def forward_propagation(self, input):
        '''
            Forward propagation convolution operation.
            Returns array of input convolved with each filter in self.
        '''
        # Unpack convolution output shape
        (conv_rows, conv_cols) = self.conv_shape(input)
        # Create 0-filled convolution array
        conv_out = np.zeros((self.n_filters, conv_rows, conv_cols))

        # Iterate over input_slice buffer
        for input_slice, row, col in self.input_slice(input):
            # Iterate over filter list
            for i in range(self.n_filters):
                # Fill convolution output array with element-wise multiplication sum
                conv_out[i, row, col] =  np.einsum('ij,ij', input_slice, self.conv_filters[i])
        return conv_out

    # TODO: Backward propagation