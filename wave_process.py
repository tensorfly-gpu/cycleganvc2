import numpy as np
import pyworld
import os
import librosa
import paddle


"""数据预处理，提取特征并保存"""
# 音频特征提取
def feature_world(wav, param):

    wav = wav.astype(np.float64) # pyworld要求输入为float64
    f0, timeaxis = pyworld.harvest(wav, param['fs'], frame_period=param['frame_period'], f0_floor=71.0, f0_ceil=800.0) # 提取基频f0
    sp = pyworld.cheaptrick(wav, f0, timeaxis, param['fs']) # 提取频谱包络sp
    ap = pyworld.d4c(wav, f0, timeaxis, param['fs']) # 提取非周期特征ap
    coded_sp = pyworld.code_spectral_envelope(sp, param['fs'], param['coded_dim']) # 对sp进行1维编码

    return f0, timeaxis, sp, ap, coded_sp


# 音频信号正则化
def wav_normlize(wav):
    max_, min_ = np.max(wav), np.min(wav)
    wav_norm = wav * (2 / (max_ - min_)) - (max_ + min_) / (max_ - min_)
    return wav_norm


def processing_wavs(file_wavs, para):
    f0s = []
    coded_sps = []

    print('开始处理!')

    for idx, file in enumerate(file_wavs):
        # 读取音频文件
        wav, _ = librosa.load(file, sr=para['fs'], mono=True)
        # 添加信号正则
        wav = wav_normlize(wav)
        # 提取world 特征,采集f0和coded_sp
        f0, _, _, _, coded_sp = feature_world(wav, para)

        if idx <= 3:
            print("processing %s" % file, '形状为', coded_sp.shape)
        elif idx == 4:
            print('对剩余语音处理中....')

        f0s.append(f0)
        coded_sps.append(coded_sp)

    print('处理完成! 共{}条语音'.format(idx + 1))

    # 计算log_f0的 均值和std
    log_f0s = np.ma.log(np.concatenate(f0s))
    log_f0s_mean = log_f0s.mean()
    log_f0s_std = log_f0s.std()

    # 计算 coded_sp 的均值和 标准差
    coded_sps_array = np.concatenate(coded_sps, axis=0)  # coded_sp的维度  T * D
    coded_sps_mean = np.mean(coded_sps_array, axis=0, keepdims=True)
    coded_sps_std = np.std(coded_sps_array, axis=0, keepdims=True)

    # 利用 coded_sp 的均值和 标准差 对特征进行正则
    coded_sps_norm = []
    for coded_sp in coded_sps:
        coded_sps_norm.append((coded_sp - coded_sps_mean) / coded_sps_std)

    return log_f0s_mean, log_f0s_std, coded_sps_mean, coded_sps_std, coded_sps_norm

#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#

"""使用训练好的模型进行音色转换"""
# 加载统计特征
def load_static(catch_path):
    static = {}
    info_f0 = np.load(os.path.join(catch_path, 'static_f0.npy'), allow_pickle=True)
    static['mean_log_f0'] = np.float64(info_f0[0])
    static['std_log_f0'] = np.float64(info_f0[1])
    info_mepc = np.load(os.path.join(catch_path, 'static_mecp.npy'), allow_pickle=True)
    static['coded_sps_mean'] = np.float64(info_mepc[0])
    static['coded_sps_std'] = np.float64(info_mepc[1])
    return static


# 转换f0
def pitch_conversion(f0, static_A, static_B):
    mean_log_f0_A = static_A["mean_log_f0"]
    std_log_f0_A = static_A["std_log_f0"]

    mean_log_f0_B = static_B["mean_log_f0"]
    std_log_f0_B = static_B["std_log_f0"]

    f0_converted = np.exp((np.ma.log(f0) - mean_log_f0_A) /
                          std_log_f0_A * std_log_f0_B + mean_log_f0_B)
    return f0_converted


# 特征归一化，和反归一化
def featu_normlize(data, data_mean, data_std, de_conver=False):
    if not de_conver:
        data_out = (data - data_mean) / data_std
    else:
        data_out = data * data_std + data_mean

    return data_out


# 音频合成
def synthesis_world(coded_sp, f0, ap, para):
    # 将coded_sp 转成 sp
    fs = para['fs']
    fftlen = pyworld.get_cheaptrick_fft_size(fs)
    decoded_sp = pyworld.decode_spectral_envelope(coded_sp, fs, fftlen)

    frame_period = para['frame_period']

    wav = pyworld.synthesize(f0, decoded_sp, ap, fs, frame_period=frame_period)
    wav = wav.astype(np.float32)
    return wav


# 加载模型对语音进行转换
def VC_model(wav, model_A2B, para, static_A, static_B):
    # 提取特征
    f0, timeaxis, sp, ap, coded_sp = feature_world(wav, para)

    # 特征截取，保证帧长是4的整数倍
    T, D = np.shape(coded_sp)
    out_T = int(T // 4 * 4)
    coded_sp = coded_sp[:out_T]
    f0 = f0[:out_T]
    ap = ap[:out_T]

    # 特征正则
    normlize_coded_sp_A = featu_normlize(coded_sp,
                                         static_A['coded_sps_mean'],
                                         static_A['coded_sps_std'],
                                         )
    # 增加维度
    normlize_coded_sp_A = np.expand_dims(normlize_coded_sp_A.T, axis=0)

    # 送入训练好的generator进行转换
    normlize_coded_sp_A = paddle.to_tensor(normlize_coded_sp_A, dtype='float32')
    model_A2B.eval()
    with paddle.no_grad():
        normlize_coded_sp_B = model_A2B(normlize_coded_sp_A).numpy()

    normlize_coded_sp_B = normlize_coded_sp_B[0]  # 取消batch的维度
    normlize_coded_sp_B = np.ascontiguousarray(normlize_coded_sp_B.T)  # T x D
    normlize_coded_sp_B = np.float64(normlize_coded_sp_B)

    # 反正则
    coded_sp_B = featu_normlize(normlize_coded_sp_B,
                                static_B['coded_sps_mean'],
                                static_B['coded_sps_std'],
                                de_conver=True)

    # 转换 f0
    f0_B = pitch_conversion(f0, static_A, static_B)

    # 将特征抓换为语音
    wav_B = synthesis_world(coded_sp_B, f0_B, ap, para)
    return wav_B