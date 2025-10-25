# -*- coding: utf-8 -*- 
# @Time : 2023/7/16 20:13 
# @Author : Yajing Zheng
# @Email: yj.zheng@pku.edu.cn
# @File : load_dat.py
import os, sys
import warnings
import glob
import yaml
import numpy as np
import path

# key-value for generate data loader according to the type of label data
LABEL_DATA_TYPE = {
    'raw': 0,
    'reconstruction': 1,
    'optical_flow': 2,
    'mono_depth_estimation': 3.1,
    'stero_depth_estimation': 3.2,
    'detection': 4,
    'tracking': 5,
    'recognition': 6
}


# generate parameters dictionary according to labeled or not
def data_parameter_dict(data_filename, label_type):
    filename = path.split_path_into_pieces(data_filename)

    if os.path.isabs(data_filename):
        file_root = data_filename
        if os.path.isdir(file_root):
            search_root = file_root
        else:
            search_root = '\\'.join(filename[0:-1])
        config_filename = path.seek_file(search_root, 'config.yaml')
    else:
        file_root = os.path.join('', 'datasets', *filename)
        config_filename = os.path.join('', 'datasets', filename[0], 'config.yaml')

    try:
        with open(config_filename, 'r', encoding='utf-8') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
    except TypeError as err:
        print("Cannot find config file" + str(err))
        raise err

    except KeyError as exception:
        print('ERROR! Task name does not exist')
        print('Task name must be in %s' % LABEL_DATA_TYPE.keys())
        raise exception

    is_labeled = configs.get('is_labeled')

    paraDict = {'spike_h': configs.get('spike_h'), 'spike_w': configs.get('spike_w')}
    paraDict['filelist'] = None

    if is_labeled:
        paraDict['labeled_data_type'] = configs.get('labeled_data_type')
        paraDict['labeled_data_suffix'] = configs.get('labeled_data_suffix')
        paraDict['label_root_list'] = None

        if os.path.isdir(file_root):
            filelist = sorted(glob.glob(file_root + '/*.dat'), key=os.path.getmtime)
            filepath = filelist[0]

            labelname = path.replace_identifier(filename, configs.get('data_field_identifier', ''),
                                                configs.get('label_field_identifier', ''))
            label_root_list = os.path.join('', 'datasets', *labelname)
            paraDict['labeled_data_dir'] = sorted(glob.glob(label_root_list + '/*.' + paraDict['labeled_data_suffix']),
                                                  key=os.path.getmtime)

            paraDict['filelist'] = filelist
            paraDict['label_root_list'] = label_root_list
        else:
            filepath = glob.glob(file_root)[0]
            rawname = filename[-1].replace('.dat', '')
            filename.pop(-1)
            filename.append(rawname)
            labelname = path.replace_identifier(filename, configs.get('data_field_identifier', ''),
                                                configs.get('label_field_identifier', ''))
            label_root = os.path.join('', 'datasets', *labelname)
            paraDict['labeled_data_dir'] = glob.glob(label_root + '.' + paraDict['labeled_data_suffix'])[0]
    else:
        filepath = file_root

    paraDict['filepath'] = filepath

    return paraDict


class SpikeStream:
    def __init__(self, **kwargs):

        self.SpikeMatrix = None
        self.filename = kwargs.get('filepath')
        if os.path.splitext(self.filename)[-1][1:] != 'dat':
            self.filename = self.filename + '.dat'
        self.spike_w = kwargs.get('spike_w')
        self.spike_h = kwargs.get('spike_h')
        if 'print_dat_detail' not in kwargs:
            self.print_dat_detail = True
        else:
            self.print_dat_detail = kwargs.get('print_dat_detail')

    def get_spike_matrix(self, flipud=True, with_head=False):

        file_reader = open(self.filename, 'rb')
        video_seq = file_reader.read()
        video_seq = np.frombuffer(video_seq, 'b')

        video_seq = np.array(video_seq).astype(np.byte)
        if self.print_dat_detail:
            print(video_seq)
        if with_head:
            decode_width = 416
        else:
            decode_width = self.spike_w
        # img_size = self.spike_height * self.spike_width
        img_size = self.spike_h * decode_width
        img_num = len(video_seq) // (img_size // 8)

        if self.print_dat_detail:
            print('loading total spikes from dat file -- spatial resolution: %d x %d, total timestamp: %d' %
                  (decode_width, self.spike_h, img_num))

        # SpikeMatrix = np.zeros([img_num, self.spike_h, self.spike_width], np.byte)

        pix_id = np.arange(0, img_num * self.spike_h * decode_width)
        pix_id = np.reshape(pix_id, (img_num, self.spike_h, decode_width))
        comparator = np.left_shift(1, np.mod(pix_id, 8))
        byte_id = pix_id // 8

        data = video_seq[byte_id]
        result = np.bitwise_and(data, comparator)
        tmp_matrix = (result == comparator)

        # if with head, delete them
        if with_head:
            delete_indx = np.arange(400, 416)
            tmp_matrix = np.delete(tmp_matrix, delete_indx, 2)

        if flipud:
            self.SpikeMatrix = tmp_matrix[:, ::-1, :]
        else:
            self.SpikeMatrix = tmp_matrix

        file_reader.close()
        self.SpikeMatrix = self.SpikeMatrix.astype(np.byte)
        return self.SpikeMatrix

        # return spikes with specified length and begin index

    def get_block_spikes(self, begin_idx, block_len=1, flipud=True, with_head=False):

        file_reader = open(self.filename, 'rb')
        video_seq = file_reader.read()
        video_seq = np.frombuffer(video_seq, 'b')

        video_seq = np.array(video_seq).astype(np.uint8)

        if with_head:
            decode_width = 416
        else:
            decode_width = self.spike_w
        # img_size = self.spike_height * self.spike_width
        img_size = self.spike_h * decode_width
        img_num = len(video_seq) // (img_size // 8)

        end_idx = begin_idx + block_len
        if end_idx > img_num:
            warnings.warn("block_len exceeding upper limit! Zeros will be padded in the end. ", ResourceWarning)
            end_idx = img_num

        if self.print_dat_detail:
            print(
                'loading total spikes from dat file -- spatial resolution: %d x %d, begin index: %d total timestamp: %d' %
                (decode_width, self.spike_h, begin_idx, block_len))

        pix_id = np.arange(0, block_len * self.spike_h * decode_width)
        pix_id = np.reshape(pix_id, (block_len, self.spike_h, decode_width))
        comparator = np.left_shift(1, np.mod(pix_id, 8))
        byte_id = pix_id // 8
        id_start = begin_idx * img_size // 8
        id_end = id_start + block_len * img_size // 8
        data = video_seq[id_start:id_end]
        data_frame = data[byte_id]
        result = np.bitwise_and(data_frame, comparator)
        tmp_matrix = (result == comparator)

        # if with head, delete them
        if with_head:
            delete_indx = np.arange(400, 416)
            tmp_matrix = np.delete(tmp_matrix, delete_indx, 2)

        if flipud:
            self.SpikeMatrix = tmp_matrix[:, ::-1, :]
        else:
            self.SpikeMatrix = tmp_matrix

        file_reader.close()
        self.SpikeMatrix = self.SpikeMatrix.astype(np.byte)
        return self.SpikeMatrix
