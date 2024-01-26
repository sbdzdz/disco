import torch

EDGE = 3
VERTEX = 2
INSIDE = 1
OUTSIDE = 0



'''


cdef    unsigned    char    point_in_polygon(np_floats[::1]    xp, np_floats[::1]    yp,    np_floats    x, np_floats    y) nogil:
    """Test relative point position to a polygon.

    Parameters
    ----------
    xp, yp : np_floats array
        Coordinates of polygon with length nr_verts.
    x, y : np_floats
        Coordinates of point.

    Returns
    -------
    c : unsigned char
        Point relative position to the polygon O: outside, 1: inside,
        2: vertex; 3: edge.

    References
    ----------
    .. [1] O'Rourke (1998), "Computational Geometry in C",
           Second Edition, Cambridge Unversity Press, Chapter 7
    """
    cdef    Py_ssize_t    i
    cdef    Py_ssize_t    nr_verts = xp.shape[0]
    cdef    np_floats    x0, x1, y0, y1, eps
    cdef    unsigned    int    l_cross = 0, r_cross = 0

    # Tolerance for vertices labelling
    eps = 1e-12

    # Initialization the loop
    x1 = xp[nr_verts - 1] - x
    y1 = yp[nr_verts - 1] - y

    # For each edge e=(i-1, i), see if it crosses ray
    for i in range(nr_verts):
        x0 = xp[i] - x
        y0 = yp[i] - y

        if (-eps < x0 < eps) and (-eps < y0 < eps):
            # it is a vertex with an eps tolerance
            return VERTEX

        # if e straddles the x-axis
        if ((y0 > 0) != (y1 > 0)):
            # check if it crosses the ray
            if ((x0 * y1 - x1 * y0) / (y1 - y0)) > 0:
                r_cross += 1
        # if reversed e straddles the x-axis
        if ((y0 < 0) != (y1 < 0)):
            # check if it crosses the ray
            if ((x0 * y1 - x1 * y0) / (y1 - y0)) < 0:
                l_cross += 1

        x1 = x0
        y1 = y0

    if (r_cross & 1) != (l_cross & 1):
        # on edge if left and right crossings not of same parity
        return EDGE

    if r_cross & 1:
        # inside if odd number of crossings
        return INSIDE

    # outside if even number of crossings
    return OUTSIDE
'''


'''


def _polygon(r, c, shape):
    """Generate coordinates of pixels inside a polygon.

    Parameters
    ----------
    r : (N,) array_like
        Row coordinates of the polygon's vertices.
    c : (N,) array_like
        Column coordinates of the polygon's vertices.
    shape : tuple, optional
        Image shape which is used to determine the maximum extent of output
        pixel coordinates. This is useful for polygons that exceed the image
        size. If None, the full extent of the polygon is used.  Must be at
        least length 2. Only the first two values are used to determine the
        extent of the input image.

    Returns
    -------
    rr, cc : ndarray of int
        Pixel coordinates of polygon.
        May be used to directly index into an array, e.g.
        ``img[rr, cc] = 1``.

    Notes
    -----
    This function ensures that `rr` and `cc` don't contain negative values.
    Pixels in the polygon with coordinates smaller than 0 are not drawn.
    """
    r = np.atleast_1d(r)
    c = np.atleast_1d(c)

    cdef Py_ssize_t minr = int(max(0, r.min()))
    cdef Py_ssize_t maxr = int(ceil(r.max()))
    cdef Py_ssize_t minc = int(max(0, c.min()))
    cdef Py_ssize_t maxc = int(ceil(c.max()))

    # make sure output coordinates do not exceed image size
    if shape is not None:
        maxr = min(shape[0] - 1, maxr)
        maxc = min(shape[1] - 1, maxc)

    # make contiguous arrays for r, c coordinates
    cdef cnp.float64_t[::1] rptr = np.ascontiguousarray(r, 'float64')
    cdef cnp.float64_t[::1] cptr = np.ascontiguousarray(c, 'float64')
    cdef cnp.float64_t r_i, c_i

    # output coordinate arrays
    rr = list()
    cc = list()

    for r_i in range(minr, maxr+1):
        for c_i in range(minc, maxc+1):
            if point_in_polygon(cptr, rptr, c_i, r_i):
                rr.append(r_i)
                cc.append(c_i)

    return np.array(rr, dtype=np.intp), np.array(cc, dtype=np.intp)
'''

def point_in_polygon(xp, yp, x, y):
    ''' xp: b x nr_verts
        yp: b x nr_verts
        x: n_points
        y: n_points

        Returns:
            b x n_points shaped mask indicating whether or
            not the point was within the polygon defined by
            :param xp, :param yp.
            Outside is indicated by the value 0, values 1-3 indicate inside or on edge/vertex.

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
    vertex_mask = (torch.any((torch.abs(xp) < eps) &
                             (torch.abs(yp) < eps), dim=1))
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
    r_cross = (r_cross.sum(dim=1) % 2 == 1).to(dtype=bool)
    l_cross = (l_cross.sum(dim=1) % 2 == 1).to(dtype=bool)
    # l_cross = (l_cross.sum(dim=1) & 1).to(dtype=bool)
    # According to Github Copilot, & 1 checks if the number is odd.

    edge_mask = r_cross != l_cross
    # wherever the left crossings have a different parity than the right crossings
    retval[edge_mask] = EDGE

    # if we have an ?even? odd? number of right crossings, the point is inside?
    inside_mask = r_cross
    # inside_mask = (r_cross.sum(dim=1) & 1)
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
    assert x.shape[0] > 0, f"Empty batch passed"
    assert y.shape[0] > 0, f"Empty batch passed"
    # nr_verts = c.shape[1]
    b = x.shape[0]
    miny = int(max(0, y.min()))
    maxy = int(torch.ceil(y.max()))
    minx = int(max(0, x.min()))
    maxx = int(torch.ceil(x.max()))

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
    w_ = len(x_i_candidates)
    h_ = len(y_i_candidates)
    point_candidates = torch.cartesian_prod(y_i_candidates, x_i_candidates)
    y_i_candidates, x_i_candidates = point_candidates[:,0] , point_candidates[:,1]
    mask = point_in_polygon(x, y, y_i_candidates, x_i_candidates)
    img_mask = torch.zeros(b, shape[0], shape[1], dtype=bool, device=x.device)
    img_mask[:, miny:maxy+1, minx:maxx+1] = mask.reshape(b, h_, w_)
    # from matplotlib import pyplot as plt
    # plt.imshow(img_mask[0, :, :])
    # plt.show()

    if return_mask:
        return img_mask
    else:
        raise NotImplementedError()
