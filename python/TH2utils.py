def fill2D(h,data1,data2,mode,weight=None):
    if mode=="normal":
        h.fill(data1,data2,weight=weight)

    return h
