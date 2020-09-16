import torch.nn as nn
import numpy as np
import os
from network.anynet import AnyNet
from option import get_args

def quantize_float(f, q):
    """Converts a float to closest non-zero int divisible by q."""
    return int(round(f / q) * q)


def adjust_ws_gs_comp(ws, bms, gs):
    """Adjusts the compatibility of widths and groups."""
    ws_bot = [int(w * b) for w, b in zip(ws, bms)]
    gs = [min(g, w_bot) for g, w_bot in zip(gs, ws_bot)]
    ws_bot = [quantize_float(w_bot, g) for w_bot, g in zip(ws_bot, gs)]
    ws = [int(w_bot / b) for w_bot, b in zip(ws_bot, bms)]
    return ws, gs


def get_stages_from_blocks(ws, rs):
    """Gets ws/ds of network at each stage from per block values."""
    ts_temp = zip(ws + [0], [0] + ws, rs + [0], [0] + rs)
    ts = [w != wp or r != rp for w, wp, r, rp in ts_temp]
    s_ws = [w for w, t in zip(ws, ts[:-1]) if t]
    s_ds = np.diff([d for d, t in zip(range(len(ts)), ts) if t]).tolist()
    return s_ws, s_ds


def generate_regnet(w_a, w_0, w_m, d, q=8):
    """Generates per block ws from RegNet parameters."""
    assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % q == 0
    ws_cont = np.arange(d) * w_a + w_0
    ks = np.round(np.log(ws_cont / w_0) / np.log(w_m))
    ws = w_0 * np.power(w_m, ks)
    ws = np.round(np.divide(ws, q)) * q
    num_stages, max_stage = len(np.unique(ws)), ks.max() + 1
    ws, ws_cont = ws.astype(int).tolist(), ws_cont.tolist()
    return ws, num_stages, max_stage, ws_cont

class RegNet(AnyNet):
    """RegNetY-1.6GF model."""
    def __init__(self, shape, num_classes=2, checkpoint_dir='checkpoint', checkpoint_name='RegNet',):
        self.shape = shape
        self.num_classes = num_classes
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name
        if len(shape) != 3:
            raise ValueError('Invalid shape: {}'.format(shape))
        self.H, self.W, self.C = shape
        
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name, 'model.pt')

        args = get_args()

        SE_ON = True
        if args.model_size == '400MF':
            DEPTH= 16
            W0= 48
            WA= 27.89
            WM= 2.09
            GROUP_W= 8
        elif args.model_size == '600MF':
            DEPTH= 15
            W0= 48
            WA= 32.54
            WM= 2.32
            GROUP_W= 16
        elif args.model_size == '800MF':
            DEPTH= 14
            W0= 56
            WA= 38.84
            WM= 2.4
            GROUP_W= 16
        elif args.model_size == '1.6GF':
            DEPTH= 27
            W0= 48
            WA= 20.71
            WM= 2.65
            GROUP_W= 24
        elif args.model_size == '3.2GF':
            DEPTH= 21
            W0= 80
            WA= 42.63
            WM= 2.66
            GROUP_W= 24
        elif args.model_size == '4.0GF':
            DEPTH= 22
            W0= 96
            WA= 31.41
            WM= 2.24
            GROUP_W= 64
        elif args.model_size == '6.4GF':
            DEPTH= 25
            W0= 112
            WA= 33.22
            WM= 2.27
            GROUP_W= 72
        elif args.model_size == '8.0GF':
            DEPTH= 17
            W0= 192
            WA= 76.82
            WM= 2.19
            GROUP_W= 56
        elif args.model_size == '12GF':
            DEPTH= 19
            W0= 168
            WA= 73.36
            WM= 2.37
            GROUP_W= 112
                                

        # Generate RegNet ws per block
        b_ws, num_s, _, _ = generate_regnet(
            w_a=WA, w_0=W0, w_m=WM, d=DEPTH
        )
        # Convert to per stage format
        ws, ds = get_stages_from_blocks(b_ws, b_ws)
        # Generate group widths and bot muls
        gws = [GROUP_W for _ in range(num_s)]
        bms = [1.0 for _ in range(num_s)]
        # Adjust the compatibility of ws and gws
        ws, gws = adjust_ws_gs_comp(ws, bms, gws)
        # Use the same stride for each stage
        ss = [2 for _ in range(num_s)]
        # Use SE for RegNetY
        se_r = 0.25 if SE_ON else None
        # Construct the model
        kwargs = {
            "stem_type": "simple_stem_in",
            "stem_w": 32,
            "block_type": "res_bottleneck_block",
            "ss": ss,
            "ds": ds,
            "ws": ws,
            "bms": bms,
            "gws": gws,
            "se_r": se_r,
            "nc": self.num_classes
        }
        super(RegNet, self).__init__(shape, num_classes, checkpoint_dir, checkpoint_name, **kwargs)