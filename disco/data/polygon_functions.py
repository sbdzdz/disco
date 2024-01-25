import torch

EDGE = 3
VERTEX = 2
INSIDE = 1
OUTSIDE = 0


def point_in_polygon(xp, yp, x, y):
    ''' xp: b x nr_verts
        yp: b x nr_verts
        x: n_points
        y: n_points

        Returns:
            b x n_points shaped mask indicating whether or
            not the point was within the polygon defined by
            :param xp, :param yp.

        '''
    eps = 1e-12
    b = xp.shape[0]
    n_points = x.shape[0]
    # center the polygon around the point
    #  want: b x nr_verts x n_points shaped tensor
    xp = xp[:, :, None] - x[None, None, :]
    yp = yp[:, :, None] - y[None, None, :]

    # tensor of "points before xp, yp"
    x1 = torch.roll(xp, 1, dims=1)
    y1 = torch.roll(yp, 1, dims=1)
    retval = torch.zeros((b, n_points), dtype=torch.uint8, device=xp.device)
    # assert torch.all(retval == OUTSIDE)

    # any polygon point very close to the point?
    vertex_mask = (torch.any(torch.abs(xp) < eps, dim=1) &
                   torch.any(torch.abs(yp) < eps, dim=1))
    retval[vertex_mask] = VERTEX

    # GPT-4:
    # This code is implementing the ray casting algorithm for determining
    # whether a point is inside a polygon. The algorithm works by casting
    # a ray from the point in both the positive and negative x-directions
    # and counting how many edges of the polygon it intersects. If the
    # counts are odd, the point is inside the polygon; if they're even,
    # it's outside.

    # So the y0 > 0 etc part checks whether the line crosses the x-axis.
    # But of those crossings, we want to count right and left crossings
    #  separately.
    # I guess this evaluates whether the point of x-axis intersection
    # (see below) corresponds to the.. endpoint? xp should be the endpoint -
    # being to the left or to the right of the y-axis ?
    # Then right_crossings would mean the end value has an x-value
    #  > 0, and left_crossings whether the end value has an x-value > 0.
    right_crossings =  ((xp * y1 - x1 * yp) / (y1 - yp)) > 0
    left_crossings =  ((xp * y1 - x1 * yp) / (y1 - yp)) < 0

    # find x-axis crossings - both the first brackets should usually evaluate
    # to the same thing except in corner cases, I think.
    r_cross = ((yp > 0) != (y1 > 0)) & right_crossings
    l_cross = ((yp < 0) != (y1 < 0)) & left_crossings

    # According to Github Copilot, & 1 checks if the number is odd.
    edge_mask = (r_cross.sum(dim=1) & 1) != (l_cross.sum(dim=1) & 1)
    # wherever the left crossings have a different parity than the right crossings
    retval[edge_mask] = EDGE

    # if we have an ?even? odd? number of right crossings, the point is inside?
    inside_mask = r_cross.sum(dim=1) & 1
    retval[inside_mask & ~edge_mask] = INSIDE

    return retval


def polygon_vectorized(y, x, shape=None, return_mask=True):
    """Generate coordinates of pixels within polygon.
        r: b x num_vertices (row coords of vertices - I guess x?)
        c: b x num_vertices (column coords of vertices)

    Parameters
    ----------
    y : (N,) ndarray
        Row coordinates of vertices of polygon.
    x : (N,) ndarray
        Column coordinates of vertices of polygon.
    shape : tuple
        Image shape which is used to determine the maximum extent of output
        pixel coordinates. This is useful for polygons that exceed the image
        size. If None, the full extent of the polygon is used.

    Returns
    -------
    Either a mask of shape b x img_shape_y x img_shape_x, or the indices obtained with
     torch.argwhere .
    """
    assert shape is not None, f"Not implemented; please pass a shape. Gains from skipping points also seem neglible."
    assert len(shape) == 2, f"I just want height and width."
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    # nr_verts = c.shape[1]
    b = x.shape[0]
    miny = int(max(0, x.min()))
    maxy = int(torch.ceil(x.max()))
    minx = int(max(0, y.min()))
    maxx = int(torch.ceil(y.max()))

    # make sure output coordinates do not exceed image size
    if shape is not None:
        maxy = min(shape[0] - 1, maxy)
        maxx = min(shape[1] - 1, maxx)
    # import numpy as np
    # make contiguous arrays for r, c coordinates
    # rptr = np.ascontiguousarray(r, 'float64')
    # cptr = np.ascontiguousarray(c, 'float64')

    # output coordinate arrays
    y_i_candidates= torch.arange(miny, maxy+1, device=x.device)
    x_i_candidates = torch.arange(minx, maxx + 1, device=x.device)
    point_candidates = torch.cartesian_prod(y_i_candidates, x_i_candidates)
    y_i_candidates, x_i_candidates = point_candidates[:,0] , point_candidates[:,1]
    mask = point_in_polygon(x, y, y_i_candidates, x_i_candidates)
    mask = mask.reshape(b, len(y_i_candidates), len(x_i_candidates))
    point_candidates = point_candidates.reshape(1, len(y_i_candidates), len(x_i_candidates))
    ids = point_candidates[mask]
    assert len(ids.shape) == 3

    if return_mask:
        whole_img_mask = torch.zeros(b, shape.shape[0], shape.shape[1], dtype=bool, device=x.device)
        whole_img_mask[ids] = True
        return whole_img_mask
    else:
        return ids
