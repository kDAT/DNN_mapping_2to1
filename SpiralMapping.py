import numpy as np


def mapping(s1, s2, delta, alpha):

    # posit_map will give us the point on the positive
    # branch of the spiral that is closest to the source point.
    pmin1, pmin2, theta_pos = posit_map(s1, s2, delta)

    # negat_map will give us the point on the negative
    # branch of the spiral that is closest to the source point.
    nmin1, nmin2, theta_neg = negat_map(s1, s2, delta)

    # check which is closer to the source point
    if (s1 - pmin1) ** 2 + (s2 - pmin2) ** 2 >= (s1 - nmin1) ** 2 + (s2 - nmin2) ** 2:
        theta_min = -theta_neg ** alpha
    else:
        theta_min = theta_pos ** alpha

    return theta_min


def posit_map(s1, s2, delta):
    # posit_map first finds a "close neighbour" for the source point
    # on the positive branch of the spiral. From this auxiliar point
    # it runs a search for the "minimum distance neighbour".

    # angle of the source coordinate
    # some adjustments are needed to make 0<=alpha<=2*pi
    if s1 >= 0 and s2 >= 0:
        alpha = np.arctan(s2 / s1)
    elif s1 < 0 and s2 >= 0:
        alpha = np.arctan(s2 / s1) + np.pi
    elif s1 < 0 and s2 < 0:
        alpha = np.arctan(s2 / s1) + np.pi
    else:
        alpha = np.arctan(s2 / s1) + 2 * np.pi

    # radius of the source point
    rad = np.sqrt(s1 ** 2 + s2 ** 2)

    # We now look for a "close neighbour" on the curve.
    # It will be assigned an angle theta on the spiral.

    # Depending on which zone the source point is,
    # we need to run the search in one direction or in the other one.
    # That's why we will use different search modes:

    # position of the source point                 |  search mode
    # -----------------------------------------------------------------
    # under the pos. branch and above neg. branch  |  inwards
    # above the pos. branch and under neg. branch  |  outwards
    # close to the origin                          |  lowCSNR

    if rad <= delta and s1 >= 0 and s2 <= 0:
        # this "if" is performed to cover a small specific area
        # of low SNR on the 4th quadrant where source signals
        # should be mapped into the pos. branch since they are not
        # close enough to the neg. branch (not straightforward to see)
        theta = 0
        mode = 'lowCSNR'
    elif rad <= delta / np.pi * alpha and alpha < np.pi:
        # all the points in the 2nd and 3rd quadrants that are below
        # the pos. branch will be considered lowCSNR
        theta = 0
        mode = 'lowCSNR'
    elif rad <= delta / np.pi * alpha:
        theta = alpha
        mode = 'inwards'
    elif rad <= delta / np.pi * (alpha + np.pi):
        theta = alpha
        mode = 'outward'
    elif rad <= delta / np.pi * (alpha + 2 * np.pi):
        theta = alpha + 2 * np.pi
        mode = 'inwards'
    elif rad <= delta / np.pi * (alpha + 3 * np.pi):
        theta = alpha + 2 * np.pi
        mode = 'outward'
    elif rad <= delta / np.pi * (alpha + 4 * np.pi):
        theta = alpha + 4 * np.pi
        mode = 'inwards'
    elif rad <= delta / np.pi * (alpha + 5 * np.pi):
        theta = alpha + 4 * np.pi
        mode = 'outward'
    elif rad <= delta / np.pi * (alpha + 6 * np.pi):
        theta = alpha + 6 * np.pi
        mode = 'inwards'
    elif rad <= delta / np.pi * (alpha + 7 * np.pi):
        theta = alpha + 6 * np.pi
        mode = 'outward'
    elif rad <= delta / np.pi * (alpha + 8 * np.pi):
        theta = alpha + 8 * np.pi
        mode = 'inwards'
    elif rad <= delta / np.pi * (alpha + 9 * np.pi):
        theta = alpha + 8 * np.pi
        mode = 'outward'
    elif rad <= delta / np.pi * (alpha + 10 * np.pi):
        theta = alpha + 10 * np.pi
        mode = 'inwards'
    else:
        theta = alpha + 10 * np.pi
        mode = 'outward'

    # coordinates of our auxiliar point "close neighbour"
    p1 = delta / np.pi * theta * np.cos(theta)
    p2 = delta / np.pi * theta * np.sin(theta)

    d = np.sqrt((p1 - s1) ** 2 + (p2 - s2) ** 2)

    # search for the "minimum distance neighbour"
    theta_min = posit_search(theta, d, mode, delta, s1, s2)
    pmin1 = delta / np.pi * theta_min * np.cos(theta_min)
    pmin2 = delta / np.pi * theta_min * np.sin(theta_min)

    return pmin1, pmin2, theta_min


def posit_search(theta, d, mode, delta, s1, s2):
    # For modes inwards and outwards the function
    # posit_search finds the "minimum distance neighbour" given
    # an initial point and a direction (inwards or outwards).
    # Search stops when finding the first minimum.

    # For mode lowCSNR the search starts always at the origin
    # and runs until a certain angle is covered. Several minimums
    # might be found but we only return the smallest one.
    inc = 1e-4  # ## Resolution of the search. Modify if needed
    if mode == 'lowCSNR':
        theta_inc = 0
        theta_min = 0

        while theta_inc <= np.pi:
            theta_inc = theta_inc + inc
            p_inc1 = delta / np.pi * theta_inc * np.cos(theta_inc)
            p_inc2 = delta / np.pi * theta_inc * np.sin(theta_inc)
            d_inc = np.sqrt((p_inc1 - s1) ** 2 + (p_inc2 - s2) ** 2)
            if d_inc < d:
                d = d_inc
                theta_min = theta_inc

    elif mode == 'inwards':
        flag = True
        theta_inc = theta
        theta_min = theta
        while flag:
            theta_inc = theta_inc - inc  # notice minus sign
            p_inc1 = delta / np.pi * theta_inc * np.cos(theta_inc)
            p_inc2 = delta / np.pi * theta_inc * np.sin(theta_inc)
            d_inc = np.sqrt((p_inc1 - s1) ** 2 + (p_inc2 - s2) ** 2)
            if d_inc > d:
                flag = False
            else:
                d = d_inc
                theta_min = theta_inc

    else:  # outwards
        flag = True
        theta_inc = theta
        theta_min = theta
        while flag:
            theta_inc = theta_inc + inc  # notice positive sign
            p_inc1 = delta / np.pi * theta_inc * np.cos(theta_inc)
            p_inc2 = delta / np.pi * theta_inc * np.sin(theta_inc)
            d_inc = np.sqrt((p_inc1 - s1) ** 2 + (p_inc2 - s2) ** 2)
            if d_inc > d:
                flag = False
            else:
                d = d_inc
                theta_min = theta_inc

    return theta_min


def negat_map(s1, s2, delta):
    # negat_map first finds a "close neighbour" for the source point
    # on the negative branch of the spiral. From this auxiliar point
    # it runs a search for the "minimum distance neighbour".

    # angle of the source coordinate
    # some adjustments are needed to make 0<=alpha<=2*pi
    if s1 >= 0 and s2 >= 0:
        alpha = np.arctan(s2 / s1) + np.pi
    elif s1 < 0 and s2 >= 0:
        alpha = np.arctan(s2 / s1) + 2 * np.pi
    elif s1 < 0 and s2 < 0:
        alpha = np.arctan(s2 / s1)
    else:
        alpha = np.arctan(s2 / s1) + np.pi

    # radius of the source point
    rad = np.sqrt(s1 ** 2 + s2 ** 2)

    # We now look for a "close neighbour" on the curve.
    # It will be assigned an angle theta on the spiral.

    # Depending on which zone the source point is,
    # we need to run the search in one direction or in the other one.
    # That's why we will use different search modes:

    # position of the source point                 |  search mode
    # -----------------------------------------------------------------
    # under the pos. branch and above neg. branch  |  inwards
    # above the pos. branch and under neg. branch  |  outwards
    # close to the origin                          |  lowCSNR

    if rad <= delta/2 and s1 <= 0 and s2 >= 0:
        # this 2nd if is performed to cover a small specific area
        # of low SNR on the 2nd quadrant where source signals
        # should be mapped into the neg. branch since they are not
        # close enough to the pos. branch (not straightforward to see)
        theta = 0
        mode = 'lowCSNR'
    elif rad <= delta / np.pi * alpha and alpha < np.pi:
        # all the points in the 3rd and 4th quadrants that are below
        # the neg. branch will be considered lowCSNR
        theta = 0
        mode = 'lowCSNR'
    elif rad <= delta / np.pi * alpha:
        theta = alpha
        mode = 'inwards'
    elif rad <= delta / np.pi * (alpha + np.pi):
        theta = alpha
        mode = 'outward'
    elif rad <= delta / np.pi * (alpha + 2 * np.pi):
        theta = alpha + 2 * np.pi
        mode = 'inwards'
    elif rad <= delta / np.pi * (alpha + 3 * np.pi):
        theta = alpha + 2 * np.pi
        mode = 'outward'
    elif rad <= delta / np.pi * (alpha + 4 * np.pi):
        theta = alpha + 4 * np.pi
        mode = 'inwards'
    elif rad <= delta / np.pi * (alpha + 5 * np.pi):
        theta = alpha + 4 * np.pi
        mode = 'outward'
    elif rad <= delta / np.pi * (alpha + 6 * np.pi):
        theta = alpha + 6 * np.pi
        mode = 'inwards'
    elif rad <= delta / np.pi * (alpha + 7 * np.pi):
        theta = alpha + 6 * np.pi
        mode = 'outward'
    elif rad <= delta / np.pi * (alpha + 8 * np.pi):
        theta = alpha + 8 * np.pi
        mode = 'inwards'
    elif rad <= delta / np.pi * (alpha + 9 * np.pi):
        theta = alpha + 8 * np.pi
        mode = 'outward'
    elif rad <= delta / np.pi * (alpha + 10 * np.pi):
        theta = alpha + 10 * np.pi
        mode = 'inwards'
    else:
        theta = alpha + 10 * np.pi
        mode = 'outward'

    # coordinates of our auxiliar point "close neighbour"
    n1 = delta / np.pi * theta * np.cos(theta + np.pi)
    n2 = delta / np.pi * theta * np.sin(theta + np.pi)

    d = np.sqrt((n1 - s1) ** 2 + (n2 - s2) ** 2)

    # search for the "minimum distance neighbour"
    theta_min = negat_search(theta, d, mode, delta, s1, s2)
    nmin1 = delta / np.pi * theta_min * np.cos(theta_min + np.pi)
    nmin2 = delta / np.pi * theta_min * np.sin(theta_min + np.pi)

    return nmin1, nmin2, theta_min


def negat_search(theta, d, mode, delta, s1, s2):
    # For modes inwards and outwards the function
    # negat_search finds the "minimum distance neighbour" given
    # an initial point and a direction (inwards or outwards).
    # Search stops when finding the first minimum.

    # For mode lowCSNR the search starts always at the origin
    # and runs until a certain angle is covered. Several minimums
    # might be found but we only return the smallest one.
    inc = 1e-4  # ## Resolution of the search. Modify if needed
    if mode == 'lowCSNR':
        theta_inc = 0
        theta_min = 0

        while theta_inc <= np.pi:
            theta_inc = theta_inc + inc
            n_inc1 = delta / np.pi * theta_inc * np.cos(theta_inc + np.pi)
            n_inc2 = delta / np.pi * theta_inc * np.sin(theta_inc + np.pi)
            d_inc = np.sqrt((n_inc1 - s1) ** 2 + (n_inc2 - s2) ** 2)
            if d_inc < d:
                d = d_inc
                theta_min = theta_inc

    elif mode == 'inwards':
        flag = True
        theta_inc = theta
        theta_min = theta
        while flag:
            theta_inc = theta_inc - inc  # notice minus sign
            n_inc1 = delta / np.pi * theta_inc * np.cos(theta_inc + np.pi)
            n_inc2 = delta / np.pi * theta_inc * np.sin(theta_inc + np.pi)
            d_inc = np.sqrt((n_inc1 - s1) ** 2 + (n_inc2 - s2) ** 2)
            if d_inc > d:
                flag = False
            else:
                d = d_inc
                theta_min = theta_inc

    else:  # outwards
        flag = True
        theta_inc = theta
        theta_min = theta
        while flag:
            theta_inc = theta_inc + inc  # notice positive sign
            n_inc1 = delta / np.pi * theta_inc * np.cos(theta_inc + np.pi)
            n_inc2 = delta / np.pi * theta_inc * np.sin(theta_inc + np.pi)
            d_inc = np.sqrt((n_inc1 - s1) ** 2 + (n_inc2 - s2) ** 2)
            if d_inc > d:
                flag = False
            else:
                d = d_inc
                theta_min = theta_inc

    return theta_min
