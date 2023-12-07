""" Script to get configured convolution, dropout, batch normalization and activation blocks.
It als returns residual connected blocks.
"""
from numpy import ndarray
from keras import backend as K
from keras.layers import (
    BatchNormalization,
    Activation,
    Dropout,
    GlobalAveragePooling2D,
    Reshape,
    Dense,
    multiply,
    Permute,
    Add,
    Conv2DTranspose,
)
from copy import deepcopy

def se_block(input_array: ndarray = None, filters: int = None, ratio: int = 8) -> ndarray:
    """ Create a squeeze-excite block. It assumes always the channel is at the end.

    Args:
        input_array: ndarray
            input image to apply feature selection using squeeze and excitation method.
        filters: int
            number of features
        ratio: int
            ratio for the squeezing

    Returns:
        ndarray: selected features with squeeze and excitation method.
    """
    init = input_array
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x

class StageConvDropoutBatchNormActivation:
    """ Apply convolution, dropout (if set true), Batch normalization (if set true), and activation (if set true).

    Returns convolved output as array

    Attributes:
        stage_input: ndarray
            output_num_features: int
                number of feature space.
        conv_config: dict
            configuration for the convolution, batch normalization, activation functions.
        skipped_input: ndarray
            skipped input to concatenate
        output_num_features: int
            required number of output features
        dimension: str
            the dimension parameter

    Methods:
        A_forward_stage() -> ndarray:
            Apply convolution, dropout, Batch normalization, and activation to the given input.
        residual_identity_block() -> ndarray:
            Residual connect by adding the shortcut and the convoluted output: x = f(x + conv_block(x))
    """
    def __init__(self, stage_input, output_num_features, conv_config, skipped_input=None, dimension=None):
        # deep copy the dictionary and mutable input parameter network_config
        self.conv_config = deepcopy(conv_config)
        self.stage_input = stage_input
        # to conv with kernel 1x1 to have the same number of channels as the preceding layer
        self.skipped_input = skipped_input
        self.output_num_features = output_num_features
        # series of conv, dropout, batch normalization, and activation operations
        self.conv_stage = []
        self.dimension = dimension  # Add dimension parameter

    def forward_stage(self) -> ndarray:
        """ Apply convolution, dropout, Batch normalization, and activation to the given input.

        Returns:
            ndarray: convolved output
        """
        current_input = self.stage_input
        current_input = self.conv_config['conv'](self.output_num_features, kernel_size=self.conv_config['kernel_size'],
                                                 strides=self.conv_config['strides'],
                                                 kernel_initializer=self.conv_config['kernel_initializer'],
                                                 use_bias=self.conv_config['use_bias'],
                                                 padding=self.conv_config['padding'])(current_input)

        if self.conv_config['apply_dropout_subblock']:
            if self.conv_config['dropout_ratio'] is None:
                self.conv_config['dropout_ratio'] = 0.3
            current_input = Dropout(self.conv_config['dropout_ratio'])(current_input)

        if self.conv_config['batch_norm']:
            current_input = BatchNormalization()(current_input)

        current_input = Activation(self.conv_config['activation'])(current_input)

        return current_input

    def residual_identity_block(self) -> ndarray:
        """ Residual connect by adding the shortcut and the convoluted output: x = f(x + conv_block(x))

        Returns:
            ndarray: residual connected and activated output array
        """
        # short_cut
        output = self.conv_config['conv'](self.output_num_features, kernel_size=1, strides=1,
                                          kernel_initializer=self.conv_config['kernel_initializer'],
                                          use_bias=self.conv_config['use_bias'], padding=self.conv_config['padding'])(
            self.skipped_input)

        # add the shortcut input with the main path output
        output = Add()([self.stage_input, output])
        output = Activation(activation=self.conv_config['activation'])(output)

        return output



class StackedConvLayerABlock:
    """
    Apply series of convolutional blocks.

    This script applies repeated convolutions for a given number of convolutions per block.
    Returns convolved, batch normalized and activated values of the input value

    Attributes:
        stage_input: ndarray
        output_num_features: int
            required number of feature channels
        conv_config: dict
            configured series of convolutions, batch normalization, dropout, and activation functions
        num_conv_per_block: int
            number of repeated blocks
        dimension: str
            the dimension parameter

    Methods:
        conv_block() -> ndarray:
            Apply series of convolutional operations.
    """
    def __init__(self, stage_input: ndarray, output_num_features: int = None, conv_config: dict = None,
            num_conv_per_block: int = None, dimension: str = None):
        self.stage_input = stage_input
        self.output_num_features = output_num_features
        self.conv_config = deepcopy(conv_config)
        self.stage_output = []
        self.dimension = dimension  # Add dimension parameter

        if num_conv_per_block is not None:
            self.conv_config['num_conv_per_block'] = num_conv_per_block

    def conv_block(self) -> ndarray:
        """
        Apply series of convolutions on the given input.

        Returns:
            ndarray: convolved, batch normalized, activated output
        """
        # self.stage_output.append(self.stage_input)
        current_output = self.stage_input

        if self.conv_config['use_residual']:
            temp = current_output
            for _ in range(self.conv_config['num_conv_per_block']):
                # temp = current_output
                current_output = StageConvDropoutBatchNormActivation(
                    stage_input=current_output,
                    output_num_features=self.output_num_features,
                    conv_config=self.conv_config,
                    dimension=self.dimension  # Pass dimension parameter
                ).forward_stage()
                '''
                Residual connection after individual convolution, batch normalization, and activation functions
                current_output = StageConvDropoutBatchNormActivation(
                    stage_input=current_output,
                    output_num_features=self.output_num_features,
                    conv_config=self.conv_config,
                    skipped_input=temp
                ).residual_identity_block()
                '''
            # residual connection at the end of the num_conv_per_block*num_conv_batch_activations
            current_output = StageConvDropoutBatchNormActivation(
                stage_input=current_output,
                output_num_features=self.output_num_features,
                conv_config=self.conv_config,
                skipped_input=temp,
                dimension=self.dimension  # Pass dimension parameter
            ).residual_identity_block()

        # no residual connection in each CONV-BATCH-ACTIVATION BLOCKS
        else:
            for _ in range(self.conv_config['num_conv_per_block']):
                current_output = StageConvDropoutBatchNormActivation(
                    stage_input=current_output,
                    output_num_features=self.output_num_features,
                    conv_config=self.conv_config,
                    dimension=self.dimension  # Pass dimension parameter
                ).forward_stage()
        return current_output

class UpConvLayer:
    """ Appy up sampling operation on the input image.

    Attributes:
    -----------
        stage_input: input array
        num_ouput_features: required number of output features
        kernel_size: size of convolutional kernel
        strides: striding operation size
        conv_upsampling: up sampling operation. '2D' for 2D network and '3D' for 3D nework

    Methods
    -------
        up_conv_layer(kernel_initializer: str = "he_normal", use_bias: bool = False, padding: str = 'same'):
        apply up sampling operation on the given input image

    """

    def __init__(self, stage_input: ndarray, num_output_features: int, kernel_size: int = None, strides: int = None,
            conv_upsampling: str = '2D'):

        if strides is None:
            self.strides = [2, 2]
        if kernel_size is None:
            self.kernel_size = [2, 2]

        if conv_upsampling == "3D":
            self.conv_upsampling = Conv3DTranspose
        else:
            self.conv_upsampling = Conv2DTranspose

        self.num_output_features = num_output_features

        self.stage_input = stage_input

    def Up_conv_layer(self, kernel_initializer: str = "he_normal", use_bias: bool = False, padding: str = 'same'):
        """ Apply up sampling on the given input.

        Args:
            kernel_initializer:  specifying kernel initializer. if it is not given by default he_normal is used.
            use_bias: Apply bias while convolution. If not specified by default 'False' is used.
            padding:  Apply padding options, e.g. 'same'

        Returns:
            Returns an up sampled value of the given input

       Notes
        -----

        Examples
        ---------
        """
        output = self.conv_upsampling(self.num_output_features, kernel_size=self.kernel_size, strides=self.strides,
                                      kernel_initializer=kernel_initializer, use_bias=use_bias, padding=padding)(
            self.stage_input)
        return output


if __name__ == '__main__':
    print("Get convolutional layers \n")
