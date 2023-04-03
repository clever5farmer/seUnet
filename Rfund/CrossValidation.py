def GetSetSize(KConst, NSize, PSize, Round):
    foldSize = [NSize//KConst, PSize//KConst]
    remainder = [NSize%KConst, PSize%KConst]
    if (KConst - Round <= remainder[0]):
        foldSize[0]+=1
    if (KConst - Round <= remainder[1]):
        foldSize[1]+=1
    return foldSize