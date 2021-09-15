import os
import logging
import datetime
import time
import math
import json
import librosa
import numpy as np
from utils import normalize

import tensorflow as tf
from tensorflow.contrib import rnn
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.cluster import KMeans
from scipy.ndimage.filters import gaussian_filter

from collections import defaultdict
from configuration import get_config
from rttm import load_rttm, Turn
from VAD_segments import VAD_chunk

config = get_config()
config.log_path = 'voxconverse-dev-embeddings.logs'
log_file = os.path.abspath(config.log_path)
logging.basicConfig(
    filename=log_file,
    level=logging.DEBUG,
    format="%(asctime)s:%(levelname)s:%(message)s"
    )
print(f'Log path: {log_file}')

data_path = '/app/datasets/voxconverse/dev/wav'
rttm_path = '/app/datasets/voxconverse/dev/rttm'
save_dir_path = '/app/voxsrc21-dia/embeddings/sequences/voxconverse-dev'
config.model_path = '/app/voxsrc21-dia/models/model.ckpt-46'
os.makedirs(save_dir_path, exist_ok=True)

def concat_segs(times, segs):
    # Concatenate continuous voiced segments
    # with segment time information (onset and offset)
    concat_seg = []
    concat_times=[]
    seg_concat = segs[0]
    seg_onset = times[0][0]
    for i in range(0, len(times)-1):
        if times[i][1] == times[i+1][0]:
            # If segments are continuous, concatenate them
            seg_concat = np.concatenate((seg_concat, segs[i+1]))
        else:
            # If not, append a new segment sequence
            concat_seg.append(seg_concat)
            seg_concat = segs[i+1]
            # Save segment time offset and append a new one
            seg_offset = times[i][1]
            seg_interval = [seg_onset, seg_offset]
            concat_times.append(seg_interval)
            seg_onset = times[i+1][0]
    else:
        concat_seg.append(seg_concat)
        # Save last time offset
        seg_offset = times[i+1][1]
        seg_interval = [seg_onset, seg_offset]
        concat_times.append(seg_interval)
        
    return concat_seg, concat_times

def get_STFTs(segs, time_segs):
    #Get 240ms STFT windows with 50% overlap, in pairs
    sr = config.sr
    STFT_windows = []
    time_windows = []
    for i, seg in enumerate(segs):
        S = librosa.core.stft(y=seg, n_fft=config.nfft, win_length=int(config.window * sr), hop_length=int(config.hop * sr))
        S = np.abs(S) ** 2
        mel_basis = librosa.filters.mel(sr=sr, n_fft=config.nfft, n_mels=40)
        # log mel spectrogram of utterances
        S = np.log10(np.dot(mel_basis, S) + 1e-6)
        
        # S.shape[1] ~= math.ceil((time_segs[i][1] - time_segs[i][0])*100+1)
        segment_time_onset = time_segs[i][0]    
        for j in range(0, S.shape[1], int(.24/config.hop)): # 0.24 / 0.01 = 24.0
            # if hop != 0.01, we can't use 12, 24, 36 frames (they stop making sense)
            # 36 frames are related to .36s of the audio
            if j + 36 < S.shape[1]:
                # in order to fit on the expected shape of the embedding network we double the window
                STFT_windows.append([S[:, j:j+24], S[:, j+12:j+36]])
                # returns the time intervals for each STFT window
                window_onset = segment_time_onset + 0.01*j
                time_windows.extend([[window_onset, window_onset+0.24], [window_onset+0.12, window_onset+0.36]])
            else:
                break
    return np.array(STFT_windows), np.array(time_windows)

def align_embeddings(embeddings, intervals):
    partitions = []
    start = 0
    end = 0
    j = 1
    for i, embedding in enumerate(embeddings):
        if (i*.12)+.24 < j*.401:
            end = end + 1
        else:
            partitions.append((start,end))
            start = end
            end = end + 1
            j += 1
    else:
        partitions.append((start,end))
    
    avg_embeddings = np.zeros((len(partitions),256))
    segment_intervals = [] 
    for i, partition in enumerate(partitions):
        avg_embeddings[i] = np.average(embeddings[partition[0]:partition[1]],axis=0)
        
        partition_interval = intervals[partition[0]:partition[1]]
        interval_onset = partition_interval[0][0]   #start of first partition
        interval_offset = partition_interval[-2][1] #end of last partition
        segment_intervals.append([interval_onset, interval_offset])
    return avg_embeddings, np.array(segment_intervals)

def getOnsets(turn):
    return turn.onset

def main():
    # Data prep
    # I'm saving only 2 embeddings i.e. first and last tisv_frames for given interval in an audio. So each .npy
    # embedding file will have a shape of (2, 256)
    tf.reset_default_graph()
    batch_size = 2 # Fixing to 2 since we take 2 for each interval #utter_batch.shape[1]
    verif = tf.placeholder(shape=[None, batch_size, 40], dtype=tf.float32)  # verification batch (time x batch x n_mel)
    batch = tf.concat([verif,], axis=1)
    # embedding lstm (3-layer default)
    with tf.variable_scope("lstm"):
        lstm_cells = [tf.contrib.rnn.LSTMCell(num_units=config.hidden, num_proj=config.proj) for i in range(config.num_layer)]
        lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells)    # make lstm op and variables
        outputs, _ = tf.nn.dynamic_rnn(cell=lstm, inputs=batch, dtype=tf.float32, time_major=True)   # for TI-VS must use dynamic rnn
        embedded = outputs[-1]                            # the last ouput is the embedded d-vector
        embedded = normalize(embedded)                    # normalize
    config_tensorflow = tf.ConfigProto(device_count = {'GPU': 0})
    saver = tf.train.Saver(var_list=tf.global_variables())
    
    all_unique_extensions = []
    # Using List as default factory
    audio_files = defaultdict(list)
    rttm_files = defaultdict(list)
    
    for audio_file in os.listdir(data_path):
        if audio_file.startswith('.'): #hidden folders
            continue;
        audio_id = os.path.splitext(audio_file)[0]
        extension = os.path.splitext(audio_file)[1]
        all_unique_extensions.append(extension)
    #     print(f'Audio id: {audio_id}')
        if extension == '.wav':
            audio_files[audio_id].append(os.path.join(data_path, audio_file))
            rttm_files[audio_id].append(os.path.join(rttm_path, audio_id + '.rttm'))
        else:
            logging.info(f'Wrong file type in {os.path.join(data_path, audio_file)}')
    
    audio_quantity = len(audio_files)
    logging.info(f'Unique file extensions: {set(all_unique_extensions)}')
    logging.info(f'Number of audios: {audio_quantity}')
    logging.info(f'Number of rttms: {len(rttm_files)}')
    
    # Extract embeddings
    # Each embedding saved file will have (2, 256)
    with tf.Session(config=config_tensorflow) as sess:
        tf.global_variables_initializer().run()
        saver.restore(sess, config.model_path)
    
        audio_count = 0
        train_sequences = np.array([]).reshape(0, 256)
    #     train_cluster_ids = []
        
        for audio_id, audio_path in audio_files.items():
            # Path: audio_files.get(audio_id)[0]
            logging.info(f'loading {audio_id} {audio_count}/{audio_quantity}')
    
            # voice activity detection            
            times, segs = VAD_chunk(2, audio_path)
            concat_seg, concat_times = concat_segs(times, segs)
            STFT_windows, time_windows = get_STFTs(concat_seg, concat_times)
            # print(len(STFT_windows), STFT_windows[0].shape)
    
            embeddings = np.array([]).reshape(0,256)
            for STFT_window in STFT_windows:
                STFT_batch = np.transpose(STFT_window, axes=(2,0,1))
                # print(STFT_batch.shape) (24, 2, 40) (240ms window * batch 2 * mels 40)
                embeddings_batch = sess.run(embedded, feed_dict={verif:STFT_batch})
                embeddings = np.concatenate((embeddings, embeddings_batch))
                
            # Turn window-level embeddings to segment-level (400ms)
            aligned_embeddings, segment_intervals = align_embeddings(embeddings, time_windows)
            
    #         # Comparar com o turns retornado pelo load_rttm para montar o train_cluster_ids
    #         turns, _, _ = load_rttm(rttm_files.get(audio_id)[0])
    #         for interval in time_windows:
    #             train_cluster_ids.append(str(speaker_count))
                
            train_sequences = np.stack((train_sequences, aligned_embeddings))
    
            audio_count += 1
            
            if (audio_count == audio_quantity or audio_count % 20 == 0):
                train_sequences_path = os.path.join(save_dir_path, f'voxcon-dev-train-sequences.npy')
                np.save(train_sequences_path, train_sequences)
                
    #             train_cluster_ids_path = os.path.join(save_dir_path, f'voxcon-dev-train-cluster-ids.npy')
    #             train_cluster_ids = np.asarray(train_cluster_ids)
    #             np.save(train_cluster_ids_path, train_cluster_ids)
                logging.info(f'saved train sequence {audio_count}/{audio_quantity}')
    
if __name__ == "__main__":
    """
    Speaker embeddings program:
    input: audio files
    output: train_sequences (x, y, 256) x sequences (utterances) of y segments of 256 observations
    """
    main()
    print('Program completed!')