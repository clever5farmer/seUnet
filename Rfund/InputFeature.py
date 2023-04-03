from enum import Enum

import numpy as np

class InputFeature(Enum):
    """
    
    """
    GRY_ = 'GRY_'   # ���O�����̍s�̂��̂ɕύX 
    EPF1 = 'EPF1'
    GAM1 = 'GAM1'
    AGC1 = 'AGC1'
    #MAH_ = 'MAH_'
    '''
    #THR = 'THR_'
    MAH_ = 'MAH_'
    GAM1 = 'GAM1'
    GAM2 = 'GAM2'
    #GAM3 = 'GAM3'
    #GAM4 = 'GAM4'
    #GAM5 = 'GAM5'
    EPF1 = 'EPF1'
    EPF2 = 'EPF2'
    #EPF3 = 'EPF3'
    #EPF4 = 'EPF4'
    #EPF5 = 'EPF5'
    AGC1 = 'AGC1'
    NML1 = 'NML1'
    NML2 = 'NML2'
    NML3 = 'NML3'
    TOP1 = 'TOP1'
    TOP2 = 'TOP2'
    TOP3 = 'TOP3'
    TOP4 = 'TOP4'
    SBLX = 'SBLX'
    SBLY = 'SBLY'
    SBLM = 'SBLM'
    SBLD = 'SBLD'
    SBL1 = 'SBL1'
    SBL2 = 'SBL2'
    SBL3 = 'SBL3'
    SBL4 = 'SBL4'
    LPL1 = 'LPL1'
    LPL2 = 'LPL2'
    MEA1 = 'MEA1'
    MEA2 = 'MEA2'
    GAU1 = 'GAU1'
    GAU2 = 'GAU2'
    MED1 = 'MED1'
    MED2 = 'MED2'
    LBP1 = 'LBP1'
    LBP2 = 'LBP2'
    LBP3 = 'LBP3'
    ETC1 = 'ETC1'
    ETC2 = 'ETC2'
    STC1 = 'STC1'
    STC2 = 'STC2'
    HGF_ = 'HGF_'
    NGP_ = 'NGP_'
    POS1 = 'POS1'
    POS2 = 'POS2'
    POS3 = 'POS3'
    SOL_ = 'SOL_'
    EMB1 = 'EMB1'
    EMB2 = 'EMB2'
    EMB3 = 'EMB3'
    KNN1 = 'KNN1'
    KNN2 = 'KNN2'
    BLT1 = 'BLT1'
    BLT2 = 'BLT2'
    #OOO_ = 'OOO_'
    '''

    def __str__(self):
        return self.name

    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented
        
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented