import numpy as np


def get_body_width(mask, y):

    row = mask[y]

    xs = np.where(row > 0.5)[0]

    if len(xs) < 2:
        return None

    return xs[-1] - xs[0]