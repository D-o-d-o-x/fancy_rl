from .base_projection_layer import BaseProjectionLayer

class IdentityProjectionLayer(BaseProjectionLayer):
    def project_from_rollouts(self, dist, rollout_data, **kwargs):
        return dist, dist
