from . import sdf
import numpy as np
# import noise

# def simplex_noise(*args, **kwargs):
#     def f(p):
#         x, y, z = p.T
#         # out = np.empty_like(x)
#         # for i in range(len(x)):
#         #     out[i] = noise.snoise3(x[i], y[i], z[i], *args, **kwargs)
#         # return out
#         out = []
#         for i in range(len(x)):
#             out.append(noise.snoise3(x[i], y[i], z[i], *args, **kwargs))
#         return np.array(out)

#     return sdf.SDF(f)