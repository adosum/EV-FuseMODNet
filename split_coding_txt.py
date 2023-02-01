import numpy as np
import os
import argparse
from PIL import Image

# from multiscaleloss import estimate_corresponding_gt_flow


parser = argparse.ArgumentParser(description='Spike Encoding')
parser.add_argument('--save-dir', type=str, default='../datasets', metavar='PARAMS',
                    help='Main Directory to save all encoding results')
# parser.add_argument('--save-env', type=str, default='indoor_flying1', metavar='PARAMS',
#                     help='Sub-Directory name to save Environment specific encoding results')
parser.add_argument('--data-path', type=str, default='.',
                    metavar='PARAMS', help='HDF5 datafile path to load raw data from')
args = parser.parse_args()

# save_path = os.path.join(args.save_dir, args.save_env)
# if not os.path.exists(save_path):
#     os.makedirs(save_path)

save_path = args.save_dir

# count_dir = os.path.join(save_path, 'count_data')
# if not os.path.exists(count_dir):
#     os.makedirs(count_dir)
#
# gray_dir = os.path.join(save_path, 'gray_data')
# if not os.path.exists(gray_dir):
#     os.makedirs(gray_dir)
#
# mask_dir = os.path.join(save_path, 'mask_data')
# if not os.path.exists(mask_dir):
#     os.makedirs(mask_dir)

event_dir = os.path.join(save_path, 'event_data')
if not os.path.exists(event_dir):
    os.makedirs(event_dir)

# gt_dir = os.path.join(save_path, 'gt_data')
# if not os.path.exists(gt_dir):
#     os.makedirs(gt_dir)


class Events(object):
    def __init__(self, num_events, width=346, height=260):
        # def __init__(self, num_events, width=640, height=480):
        self.data = np.rec.array(None, dtype=[('x', np.uint16), ('y', np.uint16), ('p', np.bool_), ('ts', np.float64)],
                                 shape=(num_events))
        self.width = width
        self.height = height

    # def generate_fimage(self, input_event=0, gray=0, mask = 0, image_raw_event_inds_temp=0, image_raw_ts_temp=0,
    #                     dt_time_temp=0):
    def generate_fimage(self, input_event=0, gray=0, image_raw_event_inds_temp=0, image_raw_ts_temp=0,
                            dt_time_temp=0):
        print(image_raw_event_inds_temp.shape, image_raw_ts_temp.shape)

        split_interval = image_raw_ts_temp.shape[0]
        data_split = 10  # N * (number of event frames from each groups)

        td_img_c = np.zeros((2, self.height, self.width, data_split), dtype=np.uint8)

        # t_index = 0

        for i in range(split_interval - (dt_time_temp - 1)):
            if image_raw_event_inds_temp[i - 1] < 0:
                frame_data = input_event[0:image_raw_event_inds_temp[i + (dt_time_temp - 1)], :]
            else:

                frame_data = input_event[
                             image_raw_event_inds_temp[i - 1]:image_raw_event_inds_temp[i + (dt_time_temp - 1)], :]

            # t_index = t_index + 1
            np.save(os.path.join(event_dir, str(i+1).zfill(4)), frame_data)
            # np.save(os.path.join(count_dir, str(i)), td_img_c)
            # np.save(os.path.join(gray_dir, str(i)), gray[i, :, :])
            # np.save(os.path.join(mask_dir, str(i)), mask[i, :, :])


# load the event data
path_image = os.path.join(args.data_path, 'images.txt')
filename = os.path.join(args.data_path, 'events.txt')
print('load data from ', filename)
f = open(filename, 'rb')
raw_data = np.loadtxt(f)
all_y = raw_data[:, 1]
all_x = raw_data[:, 2]
all_p = raw_data[:, 3]
all_ts = raw_data[:, 0]
raw_data = np.transpose(np.stack((all_y, all_x, all_ts, all_p)))


def find_closest(A, target):
    # A must be sorted
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A) - 1)
    left = A[idx - 1]
    right = A[idx]
    idx -= target - left < right - target
    return idx


# load image and mask data
imgs_dir = os.path.join(args.data_path, 'images/')
print('the image dir is', imgs_dir)
ids = [file for file in sorted(os.listdir(imgs_dir)) if not file.startswith('.')]
gray_image = np.zeros((len(ids), 375, 1242))


mask_file = os.path.join(args.data_path, 'mask/')
print('the mask dir is', mask_file)
# ids_m = [file for file in sorted(os.listdir(mask_file)) if not file.startswith('.')]
# mask = np.zeros((len(ids_m), 375, 1242))
print('there are', len(ids), 'images')
# print('there are', len(ids_m), 'masks')


image_raw_event_inds = np.zeros(len(ids))
image_raw_ts = np.zeros(len(ids) + 1)
with open(path_image) as fp:
    cnt = 0
    for line in fp:
        image_t, _ = line.split()
        image_raw_ts[cnt] = image_t
        cnt += 1
image_raw_ts = image_raw_ts[1:]

for i in range(len(image_raw_ts)):
    image_raw_event_inds[i] = find_closest(all_ts, image_raw_ts[i])

for i in range(len(ids)):
    filename = os.path.join(imgs_dir, ids[i])
    image = Image.open(filename).convert('L')
    gray_image[i, ...] = np.array(image)

# for i in range(len(ids_m)):
#     filename = os.path.join(mask_file, ids_m[i])
#     image = Image.open(filename)
#     mask[i, ...] = np.array(image)

dt_time = 1

td = Events(raw_data.shape[0])
# Events
# td.generate_fimage(input_event=raw_data, gray=gray_image, mask=mask,
#                    image_raw_event_inds_temp=image_raw_event_inds.astype(int),
#                    image_raw_ts_temp=image_raw_ts, dt_time_temp=dt_time)
td.generate_fimage(input_event=raw_data, gray=gray_image,
                   image_raw_event_inds_temp=image_raw_event_inds.astype(int),
                   image_raw_ts_temp=image_raw_ts, dt_time_temp=dt_time)

raw_data = None

print('Encoding complete!')
