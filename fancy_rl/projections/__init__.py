from .base_projection import BaseProjection
from .identity_projection import IdentityProjection
from .kl_projection import KLProjection
from .wasserstein_projection import WassersteinProjection
from .frobenius_projection import FrobeniusProjection

def get_projection(projection_name: str):
    projections = {
        "identity_projection": IdentityProjection,
        "kl_projection": KLProjection,
        "wasserstein_projection": WassersteinProjection,
        "frobenius_projection": FrobeniusProjection,
    }
    
    projection = projections.get(projection_name.lower())
    if projection is None:
        raise ValueError(f"Unknown projection: {projection_name}")
    return projection

__all__ = ["BaseProjection", "IdentityProjection", "KLProjection", "WassersteinProjection", "FrobeniusProjection", "get_projection"]