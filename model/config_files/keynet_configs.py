import numpy as np

keynet_config = {

    'KeyNet_default_config':
        {
            # Key.Net Model
            'num_filters': 8,
            'num_levels': 3,
            'kernel_size': 5,

            # Trained weights
            'weights_detector': 'model/weights/keynet_pytorch.pth',
            'weights_descriptor': 'model/HyNet/weights/HyNet_LIB.pth',

            # Extraction Parameters
            'nms_size': 15,
            'pyramid_levels': 4,
            'up_levels': 1,
            'scale_factor_levels': np.sqrt(2),
            's_mult': 22,
        },
}
