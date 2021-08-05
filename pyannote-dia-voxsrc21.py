import time
import torch

from pyannote.database import FileFinder, get_protocol

preprocessors = {'audio': FileFinder()}
protocol = get_protocol(
    'VOXSRC21.SpeakerDiarization.Challenge', preprocessors=preprocessors)

diarization_pipeline = torch.hub.load(
    'pyannote/pyannote-audio', 'dia_dihard', device='gpu')

for file in protocol.development():
    hypothesis = diarization_pipeline(file)

    uri = file['uri']
    rttmFile = '/app/datasets/voxsrc21/workshop-test/rttm/{}.rttm'.format(uri)
    with open(rttmFile, 'w') as writeRttmFile:
        hypothesis.write_rttm(writeRttmFile)

    print(f'{uri} {time.strftime("%H:%M:%S")}')
