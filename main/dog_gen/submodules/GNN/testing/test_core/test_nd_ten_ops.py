
import numpy as np
import torch

from graph_neural_networks.core import nd_ten_ops


def test_pad_right():
    pad_right_amnt = 2

    in_np = np.array([[1,2,3,4],[5,6,7,8]])
    out_np = np.array([[1,2,3,4,0,0],[5,6,7,8,0,0]])

    out_actual = nd_ten_ops.pad_right_2d(in_np, pad_right_amnt)
    np.testing.assert_almost_equal(out_np, out_actual)

    in_t = torch.tensor(in_np)
    out_t =  nd_ten_ops.pad_right_2d(in_t, pad_right_amnt).detach().cpu().numpy()
    np.testing.assert_almost_equal(out_np, out_t)


def test_pad_bottom():
    pad_right_amnt = 3

    in_np = np.array([[1,2,3,4],[5,6,7,8]])
    out_np = np.array([[1,2,3,4],[5,6,7,8],[0,0,0,0],[0,0,0,0],[0,0,0,0]])

    out_actual = nd_ten_ops.pad_bottom_2d(in_np, pad_right_amnt)
    np.testing.assert_almost_equal(out_np, out_actual)

    in_t = torch.tensor(in_np)
    out_t =  nd_ten_ops.pad_bottom_2d(in_t, pad_right_amnt).detach().cpu().numpy()
    np.testing.assert_almost_equal(out_np, out_t)
