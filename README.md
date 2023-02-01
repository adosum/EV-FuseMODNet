# EV-FuseMODNet


![Predicted flow.](res.png)  

This repository contains the Minimal code for running inference associated with EV-FuseMODNet: Moving object detection using the fusion of event camera and frame camera.
The training part will be released later.

## Introduction
Moving object detection is a crucial task for autonomous vehicles. Indeed, dynamic objects represent higher collision risk than static ones, so the trajectories of the vehicles must be planned according to the motion forecasting of the moving participants of the scene. For the traditional frame-based cameras, images can provide accumulated pixel brightness without  temporal information between them. The optical flow computation is used as the inter-frame motion information. Interestingly, event-based camera can preserve the motion information by delivering the precise timestamp of each asynchronous event data, which is more suitable for the motion analysis. Also, the event-based cameras' high temporal resolution and high dynamic range allow them to work in fast motion and extreme light scenarios. In this work, we propose a new Deep Neural Network, called EV-FuseMODNet for Moving Object Detection (MOD) that captures motion and appearance information from both event-based and frame-based cameras. The proposed method has been evaluated with the extended KittiMoSeg dataset and the generated dark KITTI sequence. An overall 27.5\% relative improvement on the extended KittiMoSeg dataset compared to the state-of-the-art approaches has been achieved. 



## Dataset

The extended KittiMoSeg dataset can be found [__**here**__](https://sites.google.com/view/fusemodnet).

Download the motion masks files from the above link and their respective raw sequence from [__**KITTI**__](https://www.cvlibs.net/datasets/kitti/).

The datasets are prepared in the `datasets/KITTI_MOD` folder:

```Shell
├── datasets
    ├── KITTI_MOD
        ├── val
            ├── dark
                ├── sequential_name1
                    ├── '****.png'
                    ├── '****.png'
                ├── sequential_name2
                    ├── '****.png'
                    ├── '****.png'
            ├── images
                ├── sequential_name
                    ├── '****.png'
                    ├── '****.png'
                    ├── event_data
                        ├── '****.npy'
                        ├── '****.npy'
                ├── 'sequential_name'
                    ├── '****.png'
                    ├── '****.png'
                    ├── event_data
                        ├── '****.npy'
                        ├── '****.npy'
            ├── mask
                ├── sequential_name1
                    ├── '****.png'
                    ├── '****.png'
                ├── sequential_name2
                    ├── '****.png'
                    ├── '****.png'
        ├── training
            ├── dark
            ├── images
            ├── mask
```
The example dataset can be downloaded from [__**here**__](https://sites.google.com/view/fusemodnet).



## Event data encoding
First following instruction in [__**event camera simulator**__](https://github.com/uzh-rpg/rpg_esim) to generate the event data, the example command are.

```
python /home/user/sim_ws/src/rpg_esim/event_camera_simulator/esim_ros/scripts/generate_stamps_file.py -i "$d" -r 30
```

```
        rosrun esim_ros esim_node \
           --data_source=2 \
           --path_to_output_bag="$d"/out.bag \
           --path_to_data_folder="$d" \
           --ros_publisher_frame_rate=60 \
           --exposure_time_ms=10.0 \
           --use_log_image=1 \
           --log_eps=0.1 \
           --contrast_threshold_pos=0.15 \
           --contrast_threshold_neg=0.15
```

Then use the following code to extract the generated ROS bag file.

```python extract_davis_bag_files.py "$d"/out.bag```

At last, split the event data according to thier timestamp.

```python split_coding_txt.py --save-dir "$d"```

## Pre-trained Model

The pretrained model can be found in [/pretrain](pretrain/) folder.




## Testing with flow visualization

The basic syntax is:

 ```
 python main.py -e --render --pretrained='checkpoint_path'
 ``` 

## Acknowledgments

Parts of this code were derived from [princeton-vl/RAFT](https://github.com/princeton-vl/RAFT) and [TimoStoff/events_contrast_maximization](https://github.com/TimoStoff/events_contrast_maximization).