import torch

import numpy as np
import random

seed=666

np.random.seed(seed)
random.seed(seed)

torch.manual_seed(seed)

torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic=True

from gumbel_vector_quantizer import GumbelVectorQuantizer


gvq = GumbelVectorQuantizer(
        dim=512,
        num_vars=320,
        temp=[2.0, 0.5, 0.999995],
        groups=2,
        combine_groups=False,
        vq_dim=256,
        time_first=True,
        weight_proj_depth=1.0,
        weight_proj_factor=1.0)


print(gvq)

x = torch.rand(8, 140, 512)

result = gvq(x, produce_targets=False)

print(result)


