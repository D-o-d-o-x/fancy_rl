try:
    import cpp_projection
except ModuleNotFoundError:
    from .base_projection_layer import ITPALExceptionLayer as KLProjectionLayer
else:
    from .kl_projection_layer import KLProjectionLayer