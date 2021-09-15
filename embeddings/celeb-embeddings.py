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

from VAD_segments import VAD_chunk

config = get_config()
config.log_path = 'voxceleb1-dev-embeddings.logs'
log_file = os.path.abspath(config.log_path)
logging.basicConfig(
    filename=log_file,
    level=logging.DEBUG,
    format="%(asctime)s:%(levelname)s:%(message)s"
    )
print(f'Log path: {log_file}')

data_path = '/app/datasets/voxceleb-1/dev/wav'
save_dir_path = '/app/voxsrc21-dia/embeddings/sequences'
config.model_path = '/app/voxsrc21-dia/models/model.ckpt-46'
os.makedirs(save_dir_path, exist_ok=True)

def concat_segs(times, segs):
    #Concatenate continuous voiced segments
    concat_seg = []
    seg_concat = segs[0]
    for i in range(0, len(times)-1):
        if times[i][1] == times[i+1][0]:
            seg_concat = np.concatenate((seg_concat, segs[i+1]))
        else:
            concat_seg.append(seg_concat)
            seg_concat = segs[i+1]
    else:
        concat_seg.append(seg_concat)
    return concat_seg

def align_embeddings(embeddings):
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
    for i, partition in enumerate(partitions):
        avg_embeddings[i] = np.average(embeddings[partition[0]:partition[1]],axis=0) 
    return avg_embeddings

def get_STFTs(segs):
    #Get 240ms STFT windows with 50% overlap, in pairs
    sr = config.sr
    STFT_windows = []
    for seg in segs:
        S = librosa.core.stft(y=seg, n_fft=config.nfft, win_length=int(config.window * sr), hop_length=int(config.hop * sr))
        S = np.abs(S) ** 2
        mel_basis = librosa.filters.mel(sr=sr, n_fft=config.nfft, n_mels=40)
        # log mel spectrogram of utterances
        S = np.log10(np.dot(mel_basis, S) + 1e-6)
        for j in range(0, S.shape[1], int(.24/config.hop)):
            if j + 36 < S.shape[1]:
                # in order to fit on the expected shape of the embedding network we double the window
                STFT_windows.append([S[:, j:j+24], S[:, j+12:j+36]])
            else:
                break
    return np.array(STFT_windows)

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
    config_tensorflow = tf.ConfigProto(device_count = {'GPU': 2})
    saver = tf.train.Saver(var_list=tf.global_variables())

    all_unique_extensions = []
    all_files = defaultdict(list)
    audio_quantity = 0
    for base_id in os.listdir(data_path):
    #     print(f'Base id: {base_id}')
        if base_id.startswith('.'): #hidden folders
            continue;
        for video_id in os.listdir(os.path.join(data_path, base_id)):
    #         print(f'Base id: {base_id} Video id: {video_id}')
            if video_id.startswith('.'): #hidden folders
                continue;
            for audio_id in os.listdir(os.path.join(data_path, base_id, video_id)):
    #             print(f'Base id: {base_id} Video id: {video_id} Audio id: {audio_id}')
                all_unique_extensions.append(os.path.splitext(audio_id)[1])
                if os.path.splitext(audio_id)[1] == '.wav':
                    # append the file path and save path to all_files
                    all_files[base_id].append(os.path.join(data_path, base_id, video_id, audio_id))
                    audio_quantity += 1
                else:
                    print(f'Wrong file type in {os.path.join(data_path, base_id, video_id, audio_id)}')
    print(f'Unique file extensions: {set(all_unique_extensions)}')
    print(f'Number of speakers: {len(all_files)}')
    print(f'Number of audios: {audio_quantity}')

    # Extract embeddings
    # Each embedding saved file will have (2, 256)
    with tf.Session(config=config_tensorflow) as sess:
        tf.global_variables_initializer().run()
        saver.restore(sess, config.model_path)

        speaker_count = 0
        total_speakers = len(all_files)
        speakers_per_batch = 50 # config.N

        batch_count = 0
        audio_count = 0
        train_sequence = np.array([]).reshape(0,256)
        train_cluster_ids = []

        for speaker_id, audio_paths in all_files.items():
            for audio_path in audio_paths:
                video_id = audio_path.split('/')[-2]
                audio_id = audio_path.split('/')[-1].replace('.wav','')

                logging.info(f'{speaker_id}-{video_id}-{audio_id} {audio_count}/{audio_quantity} batch:{batch_count}')
                # voice activity detection
                times, segs = VAD_chunk(2, audio_path)
                concat_seg = concat_segs(times, segs)
                STFT_windows = get_STFTs(concat_seg)
                # print(len(STFT_windows), STFT_windows[0].shape)

                embeddings = np.array([]).reshape(0,256)
                for STFT_window in STFT_windows:
                    STFT_batch = np.transpose(STFT_window, axes=(2,0,1))
                    # print(STFT_frames2.shape) (24, 2, 40) (240ms window * batch 2 * mels 40)
                    embeddings_batch = sess.run(embedded, feed_dict={verif:STFT_batch})
                    embeddings = np.concatenate((embeddings, embeddings_batch))

                # Turn window-level embeddings to segment-level (400ms)
                aligned_embeddings = align_embeddings(embeddings) 

                train_sequence = np.concatenate((train_sequence, aligned_embeddings))
                for embedding in aligned_embeddings:
                    train_cluster_ids.append(str(speaker_count))

                audio_count += 1

            # here: save train_sequences using stack, to separate new speaker sequence from others
            speaker_count += 1
            if (speaker_count == total_speakers or speaker_count % speakers_per_batch == 0):
                train_sequence_path = os.path.join(save_dir_path, f'vox1-train-sequences-{batch_count}.npy')
                np.save(train_sequence_path, train_sequence)

                train_cluster_ids_path = os.path.join(save_dir_path, f'vox1-train-cluster-ids-{batch_count}.npy')
                train_cluster_ids = np.asarray(train_cluster_ids)
                np.save(train_cluster_ids_path, train_cluster_ids)
                logging.info(f'saved batch {batch_count}/{math.ceil(speakers_per_batch/total_speakers)}')

                batch_count += 1
                train_sequence = np.array([]).reshape(0,256)
                train_cluster_ids = []

if __name__ == "__main__":
    """
    Speaker embeddings program:
    input: audio files
    output: npy file with shape (2, 256) [first and last tisv_frames for given interval in an audio]
    """
    main()
    print('Program completed!')
