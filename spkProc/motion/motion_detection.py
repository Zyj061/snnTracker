from utils import get_kernel, get_transform_matrix_new, visualize_images
# import kornia as tgm
import numpy as np
import torch.nn.functional as F
import torch
import os

class motion_estimation:

    def __init__(self, dvs_h, dvs_w, device, logger, debug_mode=False, debug_frame_target=200, debug_require_nonzero=True, debug_min_spike_ratio=0.0):

        self.dvs_h = dvs_h
        self.dvs_w = dvs_w
        self.device = device
        self.logger = logger

        # motion parameters
        self.orientation = range(0, 180 - 1, int(180 / 4))
        # eight moving direction
        '''
                self.ori = torch.Tensor(np.array([[-1, -1],
                    [0, -1],
                    [1, -1],
                    [-1, 0],
                    [1, 0],
                    [-1, 1],
                    [0, 1],
                    [1, 1]], dtype=np.uint8)).to(self.device)
        '''
        
        self.ori = np.array([[1, 0],
                             [1, 1],
                             [0, 1],
                             [-1, 1],
                             [-1, 0],
                             [-1, -1],
                             [0, -1],
                             [1, -1]], dtype=np.int32)
        self.speed = np.array([1, 2], dtype=np.int32)
        self.ori_x = torch.from_numpy(np.expand_dims(self.ori[:, 0], axis=1)).to(self.device).float()
        self.ori_y = torch.from_numpy(np.expand_dims(self.ori[:, 1], axis=1)).to(self.device).float()

        self.warp_matrix = get_transform_matrix_new(self.ori, self.speed, self.dvs_w, self.dvs_h, self.device)
        self.track_pre = torch.zeros(self.dvs_h, self.dvs_w)

        self.num_ori = len(self.ori)
        self.num_speed = len(self.speed)
        self.motion_pattern_num = self.num_ori * self.num_speed
        self.motion_weight = torch.ones(self.motion_pattern_num, 1, self.dvs_h, self.dvs_w) / self.motion_pattern_num
        self.tracking_threshold = 1

        self.local_pool_size = 11
        padding_width = int((self.local_pool_size - 1) / 2)
        self.pool_kernel = torch.nn.Conv2d(in_channels=1, out_channels=1,
                                           kernel_size=(self.local_pool_size, self.local_pool_size),
                                           padding=(padding_width, padding_width), bias=False)
        self.pool_kernel.weight.data = torch.ones(1, 1, self.local_pool_size, self.local_pool_size)

        self.gaussian_kernel = torch.nn.Conv2d(in_channels=1, out_channels=1,
                                               kernel_size=(self.local_pool_size, self.local_pool_size),
                                               padding=(padding_width, padding_width), bias=False)
        tmp_filter = get_kernel(self.local_pool_size, round(self.local_pool_size / 4))
        tmp_filter = tmp_filter.reshape((1, 1, self.local_pool_size, self.local_pool_size))
        self.gaussian_kernel.weight.data = torch.from_numpy(tmp_filter).float()

        # local wta inhibition size
        self.inh_size = 25
        self.padding_width = int((self.inh_size - 1) / 2)
        self.inhb_kernel = torch.nn.Conv2d(in_channels=1, out_channels=1,
                                           kernel_size=(self.inh_size, self.inh_size),
                                           padding=(self.padding_width, self.padding_width), bias=False)
        self.inhb_kernel.weight.data = torch.ones(1, 1, self.inh_size, self.inh_size)
        self.inhb_threshold = 5

        self.track_pre = self.track_pre.to(self.device)
        self.motion_weight = self.motion_weight.to(self.device)
        self.pool_kernel = self.pool_kernel.to(self.device)
        self.gaussian_kernel = self.gaussian_kernel.to(self.device)
        self.inhb_kernel = self.inhb_kernel.to(self.device)

        self._grid_cache = {}
        # Preallocate reusable buffer to reduce per-call allocations in local_wta
        self._track_voltage = torch.empty(self.num_ori, self.dvs_h, self.dvs_w, device=self.device)

        cc_motion = [[0, 255, 255],
                     [205, 95, 85],
                     [11, 134, 184],
                     [255, 255, 0],
                     [154, 250, 0],
                     [147, 20, 255],
                     [240, 32, 160],
                     [48, 48, 255]]

        cc_motion = np.transpose(np.array(cc_motion, dtype=np.float32))
        self.cc_motion = torch.from_numpy(cc_motion / 255)
        self.cc_motion = self.cc_motion.to(self.device)
        self.learning_rate = 0.1

        # Cache direction indices tensor to avoid per-call allocations in local_wta
        self._direction_indices = torch.arange(self.num_ori, device=self.device).view(-1, 1, 1)

        # Spatial downsample factors for stdp_tracking (env-controlled)
        # Set STDP_DOWNSAMPLE_H/W (>1) to enable. Example: export STDP_DOWNSAMPLE_H=2; export STDP_DOWNSAMPLE_W=2
        try:
            self.ds_h = int(os.environ.get('STDP_DOWNSAMPLE_H', '1'))
        except Exception:
            self.ds_h = 1
        try:
            self.ds_w = int(os.environ.get('STDP_DOWNSAMPLE_W', '1'))
        except Exception:
            self.ds_w = 1

        # Optional: periodically call torch.cuda.empty_cache every N calls to local_wta (0 disables)
        try:
            self.empty_cache_every = int(os.environ.get('EMPTY_CACHE_EVERY', '0'))
        except Exception:
            self.empty_cache_every = 0
        self._call_step = 0

        '''
        self.dw_ltp = torch.zeros(self.motion_pattern_num, 1, self.dvs_h, self.dvs_w)
        self.dw_ltd = torch.zeros(self.motion_pattern_num, 1, self.dvs_h, self.dvs_w)
        self.dw_ltp = self.dw_ltp.to(self.device)
        self.dw_ltd = self.dw_ltd.to(self.device)       
        '''

    def stdp_tracking(self, spikes):
        """
        参数:
            spikes: torch.Tensor
                - 形状: (H, W)
                - 类型: torch.float32 或 torch.uint8
                - 含义: 当前帧的脉冲事件矩阵，H为高度，W为宽度。每个元素为0或1，1表示该像素在当前帧有脉冲事件发生，0表示无事件。
        """
        
        prev_grad_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        # # Compute on downsampled grid to reduce cost, then upsample results back to original size
        if not torch.is_floating_point(spikes):
            spikes = spikes.float()
        use_ds = hasattr(self, 'ds_h') and hasattr(self, 'ds_w') and (self.ds_h > 1 or self.ds_w > 1)
        H_orig, W_orig = int(self.dvs_h), int(self.dvs_w)
        if use_ds:
            kh, kw = int(self.ds_h), int(self.ds_w)
            spikes_4d = torch.reshape(spikes, (1, 1, H_orig, W_orig))
            out_h = (H_orig + kh - 1) // kh
            out_w = (W_orig + kw - 1) // kw
            pad_h = out_h * kh - H_orig
            pad_w = out_w * kw - W_orig
            if pad_h > 0 or pad_w > 0:
                spikes_4d = F.pad(spikes_4d, (0, pad_w, 0, pad_h), mode='constant', value=0)
            spikes_ds_4d = F.max_pool2d(spikes_4d, kernel_size=(kh, kw), stride=(kh, kw))
            track_post = spikes_ds_4d  # 1 x 1 x H_ds x W_ds
            H2, W2 = int(track_post.shape[2]), int(track_post.shape[3])
            spikes_used_2d = torch.squeeze(track_post)
            # Downsample previous track_pre accordingly for fair comparison
            tp_4d = torch.reshape(self.track_pre, (1, 1, H_orig, W_orig)).to(track_post.device)
            if pad_h > 0 or pad_w > 0:
                tp_4d = F.pad(tp_4d, (0, pad_w, 0, pad_h), mode='constant', value=0)
            track_pre_ds_4d = F.max_pool2d(tp_4d, kernel_size=(kh, kw), stride=(kh, kw))
            track_pre_2d = torch.squeeze(track_pre_ds_4d)
        else:
            track_post = torch.reshape(spikes, (1, 1, H_orig, W_orig))
            H2, W2 = H_orig, W_orig
            spikes_used_2d = torch.squeeze(track_post)
            track_pre_2d = self.track_pre
        tmp_pool = self.pool_kernel(track_post)
        
        # Use expand to avoid real memory replication
        tmp_pool = tmp_pool.expand(self.motion_pattern_num, 1, tmp_pool.shape[-2], tmp_pool.shape[-1])

        # predict_fired变量用于存储在所有运动模式（方向×速度）下，当前帧所有激活像素点在下一个时刻的预测脉冲位置。
        # 使用批量仿射网格采样，一次性对所有运动模式进行位移预测，充分利用GPU并行。
        # 构造与当前分辨率匹配的仿射矩阵
        cache_key = (int(H2), int(W2))
        if cache_key in self._grid_cache:
            warp_theta, grid = self._grid_cache[cache_key]
        else:
            if use_ds and (H2 != self.dvs_h or W2 != self.dvs_w):
                warp_theta = get_transform_matrix_new(self.ori, self.speed, W2, H2, self.device)
            else:
                warp_theta = self.warp_matrix
            grid = F.affine_grid(warp_theta, torch.Size([self.motion_pattern_num, 1, H2, W2]), align_corners=True)
            self._grid_cache[cache_key] = (warp_theta, grid)
        # 批量复制输入到 motion_pattern_num 个通道，使用 expand 避免真实拷贝
        track_post_batched = track_post.expand(self.motion_pattern_num, -1, -1, -1)
        predict_fired = F.grid_sample(track_post_batched, grid, mode='nearest', padding_mode='zeros', align_corners=True)
        # 二值化
        predict_fired = (predict_fired > 0).to(track_post.dtype)        

        # unsqueeze函数的作用是给self.track_pre这个张量在第0维增加一个新的维度，
        # 这样原本形状为 (1, H, W) 的张量变成 (1, 1, H, W)。
        # 这里的repeat(self.motion_pattern_num, 1, 1) 是把第0维（即motion pattern的数量）扩展成self.motion_pattern_num份，
        # 结果track_pre_exp的形状变为 (self.motion_pattern_num, 1, H, W)，
        # 这样后续可以和每个运动模式的预测脉冲位置进行逐元素比较。
        track_pre_exp = track_pre_2d.unsqueeze(0).unsqueeze(1).expand(self.motion_pattern_num, 1, H2, W2)

        # STDP update the motion weight (vectorized, avoiding sparse index writes)
        tmp_bool = torch.eq(predict_fired, track_pre_exp)
        # LTP: predict == track_pre == 1
        dw_ltp = (torch.logical_and(tmp_bool, track_pre_exp == 1)).to(track_post.dtype)
        # LTD: predict == 1 and predict != track_pre
        dw_ltd = (torch.logical_and(~tmp_bool, predict_fired == 1)).to(track_post.dtype) * 2

        # Debug checkpoint: pre-pooling dw_ltp/dw_ltd stats (only during save_output_data replay)
        
        dw_ltp = self.pool_kernel(dw_ltp)
        dw_ltd = self.pool_kernel(dw_ltd)

        # dw_ltp = self.gaussian_kernel(dw_ltp)
        # dw_ltd = self.gaussian_kernel(dw_ltd)

        # dw = dw_ltp - dw_ltd
        # dw = self.gaussian_kernel(dw_ltp - dw_ltd)
        # tmp_pool[torch.where(tmp_pool == 0)] = 1
        numerator = (dw_ltp - dw_ltd)
        safe_denom = torch.where(tmp_pool != 0, tmp_pool, torch.ones_like(tmp_pool))
        quotient = numerator / safe_denom
        dw = torch.where(tmp_pool != 0, quotient, torch.zeros_like(quotient))

        # If computed on downsampled grid, upsample dw to original size before applying to motion_weight
        if use_ds:
            dw = F.interpolate(dw, size=(H_orig, W_orig), mode='nearest')
        # dw = dw / tmp_pool
        # dw = dw_ltp - dw_ltd
        with torch.no_grad():
            self.motion_weight += self.learning_rate * dw

        max_weight, _ = torch.max(self.motion_weight, dim=0)
        min_weight, _ = torch.min(self.motion_weight, dim=0)

        # Normalization
        denom = (max_weight - min_weight)
        valid_mask = denom > 0
        # Vectorized normalization across motion dimension
        self.motion_weight = torch.where(
            valid_mask,
            (self.motion_weight - min_weight) / denom,
            self.motion_weight
        )
        
        # self.motion_weight.data = F.normalize(self.motion_weight, p=2, dim=0)
        if torch.isnan(self.motion_weight).any():
            raise AssertionError("NaN detected in motion_weight")
        # self.motion_weight[torch.isinf(self.motion_weight)] = 0
        if use_ds:
            # Upsample downsampled spikes back to original size to keep track_pre shape and semantics consistent
            spikes_us = F.interpolate(track_post, size=(H_orig, W_orig), mode='nearest')
            with torch.no_grad():
                self.track_pre = torch.squeeze(spikes_us)
        else:
            with torch.no_grad():
                self.track_pre = spikes

        # del track_post, tmp_pool, predict_fired, track_pre_exp, tmp_bool, dw
        # del tmp_weight, max_weight, min_weight, spikes
        # del dw_ltd, dw_ltp
        # torch.cuda.empty_cache()

        # restore global grad setting
        torch.set_grad_enabled(prev_grad_enabled)

    def compute_motion_direction_gpu(self, dx, dy):
        """
        在GPU上直接计算运动方向，避免CPU-GPU数据传输
        替代原来的numpy.arctan2计算，消除性能瓶颈
        
        Args:
            dx: x方向运动分量 (torch.Tensor on GPU)
            dy: y方向运动分量 (torch.Tensor on GPU)
            
        Returns:
            tmp_motion: 方向编号 (0-7) (torch.Tensor on GPU)
        """
        # 使用 torch.atan2 替代 numpy.arctan2，保持在GPU上计算
        rotAng = torch.atan2(-dy, dx) * 180 / torch.pi + 180
        
        # 处理边界条件：将360度的角度归零，保证范围在[0,360)
        rotAng = torch.where(rotAng >= 360, rotAng - 360, rotAng)
        
        # 将角度均分为8个方向区间（每个区间45度），得到方向编号（0~7）
        tmp_motion = torch.floor(rotAng / (360 / 8)).long()
        
        # 确保方向编号在有效范围内 (0-7)
        tmp_motion = torch.clamp(tmp_motion, 0, 7)
        
        return tmp_motion

    def local_wta(self, spikes, timestamp, visualize=False):
        # Ensure spikes are on the correct device without unnecessary transfers
        if spikes.device != self.device:
            spikes = spikes.to(self.device, non_blocking=True)
        input_spike = torch.reshape(spikes, (1, 1, self.dvs_h, self.dvs_w))

        if False:
            # 打印self.device是CPU还是GPU
            print(f"当前设备为: {self.device} ({'GPU' if 'cuda' in str(self.device) and torch.cuda.is_available() else 'CPU'})")
            # 判断 self.motion_weight 当前是在CPU还是GPU上
            print(f"self.motion_weight 当前所在设备: { 'GPU' if 'cuda' in str(self.motion_weight.device) and torch.cuda.is_available() else 'CPU'}")
            print(f"spikes             当前所在设备: { 'GPU' if 'cuda' in str(spikes.device) and torch.cuda.is_available() else 'CPU'}")
        
        
        motion_vector_layer1 = torch.zeros(self.dvs_h, self.dvs_w, 2, dtype=torch.float32, device=self.device)
        max_w, max_wid = torch.max(self.motion_weight, dim=0)
        max_wid = torch.squeeze(max_wid)
        speedId = (max_wid % self.num_speed).detach()
        oriId = (torch.floor(max_wid / self.num_speed)).detach()

        # 这里将motion_weight的维度从 (motion_pattern_num, 1, dvs_h, dvs_w)
        # 变换为 (dvs_h, dvs_w, 1, motion_pattern_num)，
        # 方便后续reshape和与方向、速度向量的矩阵运算。
        tmp_weight = self.motion_weight.permute(2, 3, 1, 0)
        # change the dimension of matrix from (ori_num, speed_num, height, width) to (h,w, speed_num, ori_num)
        tmp_weight = torch.reshape(tmp_weight, [self.dvs_h, self.dvs_w, self.num_ori, self.num_speed])
        tmp_weight = tmp_weight.permute(0, 1, 3, 2)

        # 这两个矩阵乘法的意义：
        # tmp_weight的形状为 (dvs_h, dvs_w, num_speed, num_ori)
        # self.ori_x 和 self.ori_y 分别是方向的x分量和y分量，形状为 (num_ori, 1)
        # 下面的matmul操作相当于对每个像素、每个速度，将所有方向的权重加权到x/y分量上，实现方向分解
        # 结果tmp_weight_x, tmp_weight_y形状为 (dvs_h, dvs_w, num_speed, 1)
        tmp_weight_x = torch.matmul(tmp_weight, self.ori_x)   # 把x方向这一个维度，通过线性组合，合并掉了
        tmp_weight_y = torch.matmul(tmp_weight, self.ori_y)   # 把y方向这一个维度，通过线性组合，合并掉了
        # tmp_weight_x = torch.reshape(torch.mm(tmp_weight, self.ori_x), [self.dvs_h, self.dvs_w, self.num_speed])
        # tmp_weight_y = torch.reshape(torch.mm(tmp_weight, self.ori_y), [self.dvs_h, self.dvs_w, self.num_speed])

        max_w = torch.squeeze(max_w)
        fired_spk = torch.logical_and(spikes != 0, max_w > 0)

        tmp_weight_x = torch.mean(tmp_weight_x, dim=2)
        tmp_weight_y = torch.mean(tmp_weight_y, dim=2)
        tmp_weight_x = torch.squeeze(tmp_weight_x)
        tmp_weight_y = torch.squeeze(tmp_weight_y)

        # 🚀 GPU优化：使用稠密访问替代稀疏间接访问，提升GPU并行性能
        # 计算每个激活像素点的x/y方向运动分量，并写入motion_vector_layer1
        # 使用稠密矩阵乘法替代稀疏索引操作
        motion_vector_layer1[:, :, 0] = tmp_weight_x * fired_spk.float()
        motion_vector_layer1[:, :, 1] = tmp_weight_y * fired_spk.float()
        
        # 提取激活像素的运动分量供后续计算使用
        dx = tmp_weight_x[fired_spk]
        dy = tmp_weight_y[fired_spk]
        

        # �� GPU优化：使用GPU上的角度计算，消除CPU-GPU数据传输瓶颈
        # 对所有像素进行稠密计算，而不仅仅是激活像素
        tmp_motion_full = self.compute_motion_direction_gpu(tmp_weight_x, tmp_weight_y)
        
        # 为向后兼容，保留激活像素的方向数据
        tmp_motion = self.compute_motion_direction_gpu(dx, dy)
        
        # 🚀 GPU优化：构建方向-空间三维电压图（track_voltage），使用稠密访问
        #    形状为[num_ori, dvs_h, dvs_w]，每个像素点在其对应方向通道置1，其余为0
        track_voltage = self._track_voltage
        track_voltage.zero_()
        # 将 fired_spk 对应像素按方向填 1（scatter 方式）：
        # 索引形状: (1, H, W) 的方向索引
        tmp_motion_expanded = tmp_motion_full.view(1, self.dvs_h, self.dvs_w)
        fired_spk_float = fired_spk.float()
        # 在对应方向通道位置加上 mask（值为1）；其他为0
        track_voltage.scatter_(0, tmp_motion_expanded, fired_spk_float.unsqueeze(0))

        # 5. 卷积抑制操作
        #    - 先在方向维度插入一维（变成[方向,1,H,W]），以适配卷积核输入
        #    - 移动到设备上（如GPU）
        #    - 通过self.inhb_kernel进行局部抑制（如Winner-Take-All），再squeeze去掉多余维度
        track_voltage = torch.unsqueeze(track_voltage, 1)
        track_voltage = torch.squeeze(self.inhb_kernel(track_voltage))
        # 6. 对每个像素点，找到方向通道中最大电压值及其方向编号
        max_v, max_vid = torch.max(track_voltage, dim=0)

        # 7. 第二层激活像素筛选
        #    - 只有满足：最大电压大于阈值、原始spike激活、最大权重大于0 的像素点才被认为是有效运动点
        fired_layer2_mask = torch.logical_and(max_v >= self.inhb_threshold, torch.logical_and(spikes != 0, max_w > 0))
        # 8. 初始化最大方向编号图（max_motion）、第一层方向编号图（max_motion_layer1）、最大运动矢量图（motion_vector_max）
        max_motion = torch.zeros(self.dvs_h, self.dvs_w, dtype=torch.int64, device=self.device)
        max_motion_layer1 = torch.zeros(self.dvs_h, self.dvs_w, dtype=torch.int64, device=self.device)

        motion_vector_max = torch.zeros(self.dvs_h, self.dvs_w, 2, dtype=torch.float32, device=self.device)

        # 9. 对于第二层激活像素点，记录其最大方向编号（max_vid+1，方向编号从1开始）
        max_motion[fired_layer2_mask] = max_vid[fired_layer2_mask].detach() + 1
        # 10. 对于第一层激活像素点，记录其方向编号（tmp_motion+1，方向编号从1开始）
        # 🚀 GPU优化：tmp_motion已经是GPU tensor，无需转换
        max_motion_layer1[fired_spk] = (tmp_motion + 1).long()
        
        # 🚀 GPU优化：对于第一层激活像素点，使用稠密访问记录方向编号（从1开始）
        # 使用稠密矩阵操作替代稀疏索引，tmp_motion_full已经是GPU tensor
        # max_motion_layer1 = torch.where(fired_spk, (tmp_motion_full + 1).long(), max_motion_layer1)

        # 11. 对于未被第二层激活的像素点，将其第一层方向编号清零
        max_motion_layer1[max_motion == 0] = 0

        # 1. find the difference between m1 and mc motion
        has_layer2 = fired_layer2_mask.any()
        if has_layer2:
            tmp_vid_tensor = max_vid[fired_layer2_mask].detach()
            motion_vector_max[fired_layer2_mask] = motion_vector_layer1[fired_layer2_mask].detach()
            loser_pattern_index = torch.where(torch.logical_and(max_motion != 0, max_motion_layer1 != max_motion))
            fired2_index_x = loser_pattern_index[0]
            fired2_index_y = loser_pattern_index[1]
            voltage_block = max_v[None, None, :, :]
            voltage_block = F.pad(voltage_block, (self.padding_width, self.padding_width, self.padding_width, self.padding_width),
                                  mode='constant', value=0)
            voltage_block = F.unfold(voltage_block, (self.inh_size, self.inh_size))
            voltage_block = voltage_block.reshape([1, self.inh_size*self.inh_size, self.dvs_h, self.dvs_w])
            offset_pattern = torch.argmax(voltage_block, dim=1)
            offset_pattern = torch.squeeze(offset_pattern)
            offset_pattern_loser = offset_pattern[fired2_index_x, fired2_index_y]
            offset_x = offset_pattern_loser / self.inh_size - self.padding_width
            offset_y = torch.fmod(offset_pattern_loser, self.inh_size) - self.padding_width
            offset_x = offset_x.int()
            offset_y = offset_y.int()
            motion_vector_max[fired2_index_x, fired2_index_y, :] = motion_vector_max[fired2_index_x + offset_x,
                                                                                     fired2_index_y + offset_y, :]

        # 2. replace the loser motion pattern

        if visualize is True:
            Image_layer1 = torch.zeros(3, self.dvs_h, self.dvs_w).to(self.device)
            # 🚀 GPU优化：使用稠密访问替代稀疏索引，tmp_motion_full现在是GPU tensor
            # 对每个颜色通道进行稠密赋值
            for c in range(3):
                color_values = self.cc_motion[c, tmp_motion_full]  # 获取对应方向的颜色值
                Image_layer1[c] = torch.where(fired_spk, color_values, Image_layer1[c])

            Image_layer2 = torch.zeros(3, self.dvs_h, self.dvs_w).to(self.device)
            # 对第二层也进行类似优化，但这里tmp_vid是CPU数组，需要特殊处理
            if has_layer2:
                for c in range(3):
                    color_values = torch.zeros(self.dvs_h, self.dvs_w, device=self.device)
                    color_values[fired_layer2_mask] = self.cc_motion[c, tmp_vid_tensor]
                    Image_layer2[c] = torch.where(fired_layer2_mask, color_values, Image_layer2[c])

            self.logger.add_image('motion_estimation/M1 estimation', Image_layer1, timestamp)
            self.logger.add_image('motion_estimation/MC estimation', Image_layer2, timestamp)

        # track_voltage.to(self.device_cpu)

        del dx, dy, tmp_motion, tmp_motion_full
        # Optional periodic empty_cache to mitigate allocator stalls in debug/replay; disabled by default
        if self.empty_cache_every > 0:
            self._call_step += 1
            if (self._call_step % self.empty_cache_every) == 0:
                torch.cuda.empty_cache()
 
        return max_motion, motion_vector_max, motion_vector_layer1
