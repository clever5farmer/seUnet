from enum import Enum


class Augmentation(Enum):
    """
    Type list of augmentation
    """
    ORIGINAL = 'Original'
    #AFFINE = 'Affine'
    #SCALEX = 'ScaleX'
    #SCALEY = 'ScaleY'
    #TRANSLATION20 = 'Translation20'
    #TRANSLATION40 = 'Translation40'
    #CUTOUT = 'Cutout'
    #CUTOUT2 = 'Cutout2'
    #INVERT = 'Invert'

    def __str__(self):
        return self.name