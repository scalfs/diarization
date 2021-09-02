from tqdm import tqdm
import glob
import subprocess


files = glob.glob('/home/jovyan/work/datasets/voxceleb-2/dev/aac/*/*/*.m4a')
files.sort()

print('Converting files from AAC to WAV')
for fname in tqdm(files):
    outfile = fname.replace('.m4a', '.wav')
    out = subprocess.call(
        'ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s >/dev/null 2>/dev/null' % (fname, outfile), shell=True)
    if out != 0:
        raise ValueError('Conversion failed %s.' % fname)

for fname in tqdm(files):
    out = subprocess.call('rm -rf %s' % (fname), shell=True)
    if out != 0:
        raise ValueError('Removal error %s.' % fname)
