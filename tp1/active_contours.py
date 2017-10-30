import numpy as np
import math
from tp1 import image_operations as ops
from tp1 import umbralization as umb
from basics import transforms as tr

object_value = -3
empty_value = 3
lin_value = -1
lout_value = 1
deltas = [(1, 0), (-1, 0), (0, 1), (0, -1)]

lin_color = [255, 255, 255]
lout_color = [0, 255, 0]

def _set_px(img, i, j, value):
    for k in range(3):
        img[i, j, k] = value

def _get(m, i, j):
    if i < 0 or i >= len(m) or j < 0 or j >= len(m[0]):
        return 0
    return m[i, j]

def _is_inside(up_left, down_right, x, y):
    if x < up_left[0] or x > down_right[0]:
        return False
    if y < up_left[1] or y > down_right[1]:
        return False
    return True

def _color_averages(img, contours):
    in_color = [0, 0, 0]
    out_color = [0, 0, 0]
    in_count = 0
    out_count = 0

    for i in range(0, len(img)):
        for j in range(0, len(img[0])):
            if contours[i, j] < 0:
                in_color += img[i, j]
                in_count += 1
            else:
                out_color += img[i, j]
                out_count += 1

    return in_color / in_count, out_color / out_count

def norm(x, y):
    if len(x) != len(y):
        raise AssertionError
    total = 0
    for i in range(len(x)):
        diff = x[i] - y[i]
        total += math.pow(diff, 2)
    return math.sqrt(total)

def _should_be_in(pixel, in_color, out_color):
    in_value = norm(pixel, in_color)
    out_value = norm(pixel, out_color)
    return in_value < out_value

def _is_object_px(contours, x, y):
    for delta in deltas:
        xx, yy = x + delta[0], y + delta[1]
        if _get(contours, xx, yy) > 0:
            return False
    return True

def _is_background_px(contours, x, y):
    for delta in deltas:
        xx, yy = x + delta[0], y + delta[1]
        if _get(contours, xx, yy) < 0:
            return False
    return True

def _generate_contours(img, start, end):
    contours = np.zeros(shape=(len(img), len(img[0])))
    lin = []
    lout = []
    # Fill with initial rectangle values
    for i in range(0, len(img)):
        for j in range(0, len(img[0])):
            if i == start[0] or i == end[0] or j == start[1] or j == end[1]:
                contours[i, j] = lin_value
                lin.append((i, j))
            elif i == start[0] - 1 or i == end[0] + 1 or j == start[1] - 1 or j == end[1] + 1:
                contours[i, j] = lout_value
                lout.append((i, j))
            elif _is_inside(start, end, i, j):
                contours[i, j] = object_value
            else:
                contours[i, j] = empty_value
    return contours, lin, lout

# contours is mutated, lin and lout are not, new instances are returned
def _adjust_contour(img, contours, lin, lout, out_change_fx, in_change_fx):
    change_made = False
    new_lout = []
    for out in lout:
        x, y = out
        if out_change_fx(x, y):
            change_made = True
            contours[x, y] = lin_value
            lin.append((x, y))
            for delta in deltas:
                xx, yy = x + delta[0], y + delta[1]
                if _get(contours, xx, yy) == empty_value:
                    contours[xx, yy] = lout_value
                    new_lout.append((xx, yy))
        else:
            new_lout.append(out)
    lout = new_lout
    new_lin = []
    for in_v in lin:
        x, y = in_v
        if _is_object_px(contours, x, y):
            contours[x, y] = object_value
        else:
            if in_change_fx(x, y):
                new_lin.append(in_v)
            else:
                change_made = True
                contours[x, y] = lout_value
                lout.append((x, y))
                for delta in deltas:
                    xx, yy = x + delta[0], y + delta[1]
                    if _get(contours, xx, yy) == object_value:
                        contours[xx, yy] = lin_value
                        new_lin.append((xx, yy))
    lin = new_lin
    new_lout = []
    for out in lout:
        x, y = out
        if _is_background_px(contours, x, y):
            contours[x, y] = empty_value
        else:
            new_lout.append(out)
    lout = new_lout
    return lin, lout, change_made

def draw_contours(img, lin, lout):
    for in_v in lin:
        i, j = in_v
        img[i, j] = lin_color
    for out in lout:
        i, j = out
        img[i, j] = lout_color

def active_contours_rect(img, rect=((200, 115), (250, 150))):
    start = rect[0]
    end = rect[1]
    [contours, lin, lout] = _generate_contours(img, start, end)
    return active_contours(img, contours, lin, lout)

def active_contours(img, contours, lin, lout):
    max_cycles = min(len(img), len(img[0]))
    continue_ = True
    cycles_done = 0
    while cycles_done < max_cycles and continue_:
        print(cycles_done)
        cycles_done += 1
        [in_avg, out_avg] = _color_averages(img, contours)
        change_fx = lambda x, y: _should_be_in(img[x, y], in_avg, out_avg)
        [lin, lout, continue_] = _adjust_contour(img, contours, lin, lout, change_fx, change_fx)

    # ???
    # gauss_filter = ops.apply_gauss_filter(contours, sigma=1, size=5)
    # g_out_change_fx = lambda x, y: gauss_filter[x, y] * contours[x, y] < 0
    # g_in_change_fx = lambda x, y: gauss_filter[x, y] * contours[x, y] > 0
    # _adjust_contour(img, contours, lin, lout, g_out_change_fx, g_in_change_fx)

    draw_contours(img, lin, lout)
    return img, contours, lin, lout