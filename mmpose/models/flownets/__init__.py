from .raft import RAFT
from .FlowFormer import FlowFormer
from .smurf import SMURF
# from .gaflow import GAFlow

# @MODELS.register_module()
# class FlowFormer(nn.Module):
#     def __init__(self, args_dict):
#         super(FlowFormer, self).__init__()
#         self.args = Namespace(**args_dict)

#         self.flowformer = build_flowformer(self.args)

#     def forward(self, x):
#         return self.flowformer(x)

__all__ = ['RAFT', 'FlowFormer', 'SMURF']