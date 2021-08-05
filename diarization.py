import time

import torch
from IPython.display import Audio

from pyannote.database import FileFinder, get_protocol
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

preprocessors = {'audio': FileFinder()}
protocol = get_protocol('VOXCON.SpeakerDiarization.Sample', preprocessors=preprocessors)

diarization_pipeline = torch.hub.load('pyannote/pyannote-audio', 'dia_dihard')

ders = []
jers = []
hypotheses = []

derMetric = DiarizationErrorRate(collar=0.25)
jerMetric = JaccardErrorRate(collar=0.25)

for file in protocol.test():
    hypothesis = diarization_pipeline(file)
    hypotheses.append(hypothesis)
    
    reference = file["annotation"]
    uem = file['annotated']
    der = derMetric(reference, hypothesis, uem)['diarization error rate']
    jer = jerMetric(reference, hypothesis, uem)['diarization error rate']
    ders.append(der)
    jers.append(jer)
    
    uri = file['uri']
    print(f'{uri} DER = {100 * der:.1f}% JER = {100 * jer:.1f}% {time.strftime("%H:%M:%S")}')

# der['diarization error rate'] = (der['false alarm'] + der['missed detection'] + der['confusion']) / der['total']

metric = DiarizationErrorRate(collar=0.25)

i = 0
for file in protocol.test():
    hypotesis = hypotheses[i]
    reference = file["annotation"]
    uem = file['annotated']
    der = metric(reference, hypotesis)
    print(der)
    i+=1


# Compute global diarization error rate and confidence interval

global_value = abs(metric)
mean, (lower, upper) = metric.confidence_interval()

print(f'DER_total = {100 * global_value:.1f}% mean = {100 * mean:.1f}%')
print(f'lower = {100 * lower:.1f}% upper = {100 * upper:.1f}%')

# for file in protocol.test():
#     der = derMetric(groundtruth, diarization)
