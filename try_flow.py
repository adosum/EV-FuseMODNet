# import cv2
#
# from vis_utils import draw_color_wheel_np, flow2img
#
# flow = draw_color_wheel_np(256, 256)
# cv2.imwrite('circleflow.png', flow)
import cv2
# def process(n, cur, rest, p):
#     if rest == 0:
#         return 1 if cur == p else 0
#     if cur == 1:
#         return process(n, 2, rest - 1, p)
#     if cur == n:
#         return process(n, n - 1, rest - 1, p)
#     return process(n, cur - 1, rest - 1, p) + process(n, cur + 1, rest - 1, p)
#
#
# def main(n, m, k, p):
#     return process(n, m, k, p)
#
#
# if __name__ == '__main__':
#     print(main(7, 2, 5, 3))
import numpy as np
from vis_utils import draw_mask
img = np.full((10, 10, 3), 128, np.uint8)

# sample mask
mask = np.zeros((10, 10), np.uint8)
mask[3:6, 3:6] = 1
out = draw_mask(img,mask)
cv2.imshow('', out)
cv2.waitKey(-1)