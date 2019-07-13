import torch.nn as nn
import torch
from . import image_processing


class ImageDegradationPipeline(nn.Module):

    def __init__(self, configs):
        """ Image Degradation Pipeline.

        Args:
            configs: list of modules to be implemented and their parameters.
                     The list should contain tuple of a form (str, dict),
                     where str indicate module class name (see
                     image_processing.py), and dict contain the key-value of
                     the parameter of such module.
        """
        super().__init__()
        self.initialize_pipeline(configs)

    def initialize_pipeline(self, configs):
        pipeline = []
        # initialize module.
        for c in configs:
            class_ = getattr(image_processing, c[0])
            module = class_(**c[1])
            pipeline.append(module)
        self._pipeline = nn.Sequential(*pipeline)
        # self._pipeline = tuple(pipeline)
    def forward(self, image):
        # import torchvision.transforms as transforms
        # trans = transforms.ToPILImage()
        # for index, func in enumerate(self._pipeline):
        #     image = func(image)
        #     # save images
        #     # image_trans = trans((torch.clamp(image, 0.0, 1.0)).squeeze())
        #     # image_trans.save('./train_images/tmp_{:02d}.png'.format(index), quality=100)
        # return image
        return self._pipeline(image)

