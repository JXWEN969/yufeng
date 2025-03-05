_base_ = [
    '../_base_/models/fcos_r50_caffe_fpn_4x4_1x_coco.py',
    '../_base_/datasets/didi_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
