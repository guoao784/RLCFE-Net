try:
    import swattention
    from .TransNeXt.TransNext_cuda import *
except ImportError as e:
    from .TransNeXt.TransNext_native import *
    pass

