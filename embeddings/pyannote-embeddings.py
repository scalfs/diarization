import time
import torch
from pyannote.database import get_protocol, FileFinder

emb = torch.hub.load('pyannote/pyannote-audio', 'emb')
sad = torch.hub.load('pyannote/pyannote-audio', 'sad')

save_dir_path = '/app/voxsrc21-dia/embeddings/sequences'

preprocessors = {'audio': FileFinder()}
protocol = get_protocol('VOXSRC21.SpeakerDiarization.Challenge', preprocessors=preprocessors)

for idx, file in enumerate(protocol.test()):
    uri = file['uri']
    embeddings = emb(file)
    speech_detection = sad(file) 
    print(uri, time.strftime("%H:%M:%S"))
          

    annotation = Annotation()
    annotation.uri = sample_id
    for jdx, speaker_id in enumerate(labels):
        segment_interval = intervals[idx][jdx]
        annotation[Segment(segment_interval[0],
                           segment_interval[1])] = speaker_id

    rttm_file = '{}/{}.rttm'.format(rttm_dir, sample_id)
    with open(rttm_file, 'w') as file:
        annotation.support().write_rttm(file)
        
    train_sequences_path = os.path.join(save_dir_path, f'voxcon-dev-sequences.npy')
    np.save(train_sequences_path, train_sequences)

    intervals_path = os.path.join(save_dir_path, f'voxcon-dev-intervals.npy')
    np.save(intervals_path, sequence_intervals)

    # rttm_file_collar = '{}/rttm_colar/{}.rttm'.format(rttm_dir, sample_id)
    # with open(rttm_file_collar, 'w') as file:
    #     annotation.support(0.481).write_rttm(file)