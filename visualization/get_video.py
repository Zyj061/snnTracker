# -*- coding: utf-8 -*- 
# @Time : 2022/6/12 15:21 
# @Author : Yajing Zheng
# @File : visualize.py
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import matplotlib.cm as cm
import matplotlib.animation as animation

def obtain_spike_video(spikes, video_filename, **dataDict):
    spike_h = dataDict.get('spike_h')
    spike_w = dataDict.get('spike_w')
    timestamps = spikes.shape[0]

    mov = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'MJPG'), 30, (spike_w, spike_h))

    for iSpk in range(timestamps):
        tmpSpk = spikes[iSpk, :, :] * 255
        tmpSpk = cv2.cvtColor(tmpSpk.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        mov.write(tmpSpk)

    mov.release()


def obtain_reconstruction_video(images, video_filename, **dataDict):
    spike_h = dataDict.get('spike_h')
    spike_w = dataDict.get('spike_w')

    img_num = images.shape[0]
    mov = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'MJPG'), 30, (spike_w, spike_h))
    for iImg in range(img_num):
        tmp_img = images[iImg, :, :]
        tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_GRAY2BGR)
        mov.write(tmp_img)

    mov.release()


def obtain_mot_video(spikes, video_filename, res_filepath, **dataDict):
    spike_h = dataDict.get('spike_h')
    spike_w = dataDict.get('spike_w')

    gt_file = dataDict.get('labeled_data_dir')
    gt_boxes = {}
    if gt_file is not None:
        gt_f = open(gt_file, 'r')
        gt_lines = gt_f.readlines()
        for line in gt_lines:
            gt_term = line.split(',')
            time_step = gt_term[0]
            box_id = gt_term[1]
            x = float(gt_term[2])
            y = float(gt_term[3])
            w = float(gt_term[4])
            h = float(gt_term[5])

            if str(time_step) not in gt_boxes:
                gt_boxes[str(time_step)] = []
            bbox = [box_id, x, y, w, h]
            gt_boxes[str(time_step)].append(bbox)

        gt_f.close()

    result_file = res_filepath
    test_boxes = {}
    result_f = open(result_file, 'r')
    result_lines = result_f.readlines()
    color_dict = {}

    for line in result_lines:
        res_box = line.split(',')
        time_step = res_box[0]
        track_id = res_box[1]
        if track_id not in color_dict.keys():
            colors = (np.random.rand(1, 3) * 255).astype(np.uint8)
            color_dict[track_id] = np.squeeze(colors)

        x = float(res_box[2])
        y = float(res_box[3])
        w = float(res_box[4])
        h = float(res_box[5])

        if str(time_step) not in test_boxes:
            test_boxes[str(time_step)] = []

        test_box = [track_id, x, y, w, h]
        test_boxes[str(time_step)].append(test_box)

    result_f.close()

    mov = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'MJPG'), 30, (spike_w, spike_h))

    timestamps = spikes.shape[0]
    for t in range(151, timestamps):
    # for t in range(160, 1000):
        tmp_ivs = spikes[t, :, :] * 255
        tmp_ivs = cv2.cvtColor(tmp_ivs.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        if len(gt_boxes) > 0:
            if str(t) in gt_boxes:
                gts = gt_boxes[str(t)]
                gt_num = len(gts)
                for i in range(gt_num):
                    box = gts[i]
                    box_id = box[0]
                    cv2.rectangle(tmp_ivs, (int(box[2]), int(box[1])),
                                  (int(box[2] + box[4]), int(box[1] + box[3])),
                                  (int(255), int(255), int(255)), 2)

        if str(t) in test_boxes:
            test = test_boxes[str(t)]
            test_num = len(test)
            for i in range(test_num):
                box = test[i]
                box_id = box[0]
                colors = color_dict[box_id]
                cv2.rectangle(tmp_ivs, (int(box[2]), int(box[1])),
                              (int(box[2] + box[4]), int(box[1] + box[3])),
                              (int(colors[0]), int(colors[1]), int(colors[2])), 2)

        mov.write(tmp_ivs)

    mov.release()


def obtain_detection_video(spikes, video_filename, res_filepath, evaluate_seq_len, begin_idx=0, **dataDict):
    spike_h = dataDict.get('spike_h')
    spike_w = dataDict.get('spike_w')

    gt_file = dataDict.get('labeled_data_dir')
    gt_boxes = {}
    if gt_file is not None:
        start_idx = begin_idx
        end_idx = begin_idx + evaluate_seq_len
        for seq_no in range(start_idx, end_idx):
            gt_filename = gt_file[seq_no]
            gt_f = open(gt_filename, 'r')

            gt_lines = gt_f.readlines()
            for line in gt_lines:
                tmp_box = line.split(',')

                x = float(tmp_box[0])
                y = float(tmp_box[1])
                w = float(tmp_box[2])
                h = float(tmp_box[3])
                box_id = int(0)

                if str(seq_no) not in gt_boxes:
                    gt_boxes[str(seq_no)] = []
                bbox = [box_id, x, y, w, h]
                gt_boxes[str(seq_no)].append(bbox)

            gt_f.close()

    result_file = res_filepath
    test_boxes = {}
    result_f = open(result_file, 'r')
    result_lines = result_f.readlines()
    color_dict = {}

    for line in result_lines:
        res_box = line.split(',')
        time_step = res_box[0]
        track_id = res_box[1]
        if track_id not in color_dict.keys():
            colors = (np.random.rand(1, 3) * 255).astype(np.uint8)
            color_dict[track_id] = np.squeeze(colors)

        x = float(res_box[2])
        y = float(res_box[3])
        w = float(res_box[4])
        h = float(res_box[5])

        if str(time_step) not in test_boxes:
            test_boxes[str(time_step)] = []

        test_box = [track_id, x, y, w, h]
        test_boxes[str(time_step)].append(test_box)

    result_f.close()

    mov = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'MJPG'), 30, (spike_w, spike_h))

    block_len = spikes.shape[0]
    # gt_intv = int(block_len/evaluate_seq_len)
    gt_intv = 400

    # for t in range(150, block_len):
    for i_gt in range(start_idx+1, end_idx):
        t = i_gt * gt_intv + int(gt_intv/2)
        tmp_ivs = spikes[t, :, :] * 255
        tmp_ivs = cv2.cvtColor(tmp_ivs.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        if len(gt_boxes) > 0:
            gts = gt_boxes[str(i_gt)]
            gt_num = len(gts)
            for i in range(gt_num):
                box = gts[i]
                cv2.rectangle(tmp_ivs, (int(spike_w - box[1]), int(box[2])),
                              (int(spike_w - box[1] - box[3]), int(box[2] + box[4])),
                              (int(255), int(255), int(255)), 2)

        if str(t) in test_boxes:
            test = test_boxes[str(t)]
            test_num = len(test)
            for i in range(test_num):
                box = test[i]
                box_id = box[0]
                colors = color_dict[box_id]
                cv2.rectangle(tmp_ivs, (int(box[2]), int(box[1])),
                              (int(box[2] + box[4]), int(box[1] + box[3])),
                              (int(colors[0]), int(colors[1]), int(colors[2])), 2)

        mov.write(tmp_ivs)

    mov.release()

def get_heatVideo(results, video_filename):
    results = np.array(results)
    frame_num = results.shape[0]
    frames = []

    fig = plt.figure()
    for i in range(frame_num):
        tmp_res = results[i]
        # frames.append([plt.imshow(tmp_res, cmap=cm.Greys_r, animated=True)])
        frames.append([plt.imshow(tmp_res, cmap=cm.Blues, animated=True)])

    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
                                    repeat_delay=1000)

    # change the path to where you save ffmpeg
    plt.rcParams['animation.ffmpeg_path'] = 'F:\\ffmpeg-N-99818-g993429cfb4-win64-gpl-shared-vulkan\\bin\\ffmpeg.exe'
    FFwrite = animation.FFMpegWriter(fps=30, extra_args=['-vcodec', 'libx264'])
    ani.save(video_filename, writer=FFwrite)
    plt.show()




