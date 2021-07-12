import scipy.io
from scipy.io import wavfile
import os
import librosa
import soundfile as sf
import numpy as np
# define o path dos arquivos .wav contendo os rotulos
path = "/home/gabriel/Desktop/My_tasks/IA-DEEPLEARNING/IC/Codigos/data_vale/audio"

# define o path dos arquivos .txt contendo os rotulos
path_rotulo = "/home/gabriel/Desktop/My_tasks/IA-DEEPLEARNING/IC/Codigos/data_vale/label"

# cria o arquivo lst onde cada linha contém o nome do arquivo a ser lido
lst_file_train = open("Cambio.train.lst", 'w')
lst_file_val = open("Cambio.development.lst", 'w')
lst_file_test = open("Cambio.test.lst", 'w')

# cria o arquivo uem onde define o intervalo de audio anotado
# ou seja o intervalo do arquivo de audio foi rotulado

uem_file_train = open("Cambio.train.uem", 'w')
uem_file_val = open("Cambio.development.uem", 'w')
uem_file_test = open("Cambio.test.uem", 'w')

# escreve o nome dos arquivos disponíveis no lst file
# escreve o arquivo uem, onde contém a faixa de tempo de rotulação

#  uem file : {uri} 1 start end
# lst file  : {uri}

# auxiliares de divisão de treino, validação e teste
qtd_de_audio = len(os.listdir(path))
qtd_treino = np.int(0.8*qtd_de_audio)
qtd_val = np.int(0.1*qtd_de_audio)
qtd_test = qtd_de_audio-qtd_val-qtd_treino
qtd_atual = 1


# randomizando os arquivos
audios = np.array(os.listdir(path))
idx = np.arange(qtd_de_audio)
np.random.shuffle(idx)
idx_train = idx[:qtd_treino]
idx_val = idx[qtd_treino:qtd_treino+qtd_val]
idx_test = idx[qtd_val+qtd_treino:]


# audios de treino
audios_treino = audios[idx_train]
# audios de validação
audios_val = audios[idx_val]
# audios de teste
audios_test = audios[idx_test]

# for para .lst e .uem de treino
for name in audios_treino:
    audio, sr = sf.read(f'{path}/{name}')
    duration = librosa.core.get_duration(audio, sr)
    lst_file_train.write(f'{name[:-4]}\n')
    uem_file_train.write(f'{name[:-4]} 1 0.000 {duration}\n')


# for para .lst e .uem de validação
for name in audios_val:
    audio, sr = sf.read(f'{path}/{name}')
    duration = librosa.core.get_duration(audio, sr)
    lst_file_val.write(f'{name[:-4]}\n')
    uem_file_val.write(f'{name[:-4]} 1 0.000 {duration}\n')


# for para .lst e .uem de teste
for name in audios_test:
    audio, sr = sf.read(f'{path}/{name}')
    duration = librosa.core.get_duration(audio, sr)
    lst_file_test.write(f'{name[:-4]}\n')
    uem_file_test.write(f'{name[:-4]} 1 0.000 {duration}\n')


# fechando o arquivo lst
lst_file_train.close()
lst_file_test.close()
lst_file_val.close()
# fechando o arquivo uem
uem_file_train.close()
uem_file_test.close()
uem_file_val.close()

# criando o RTTM para o treino

rttm_train = open("Cambio.train.rttm", "w")

for name in audios_treino:
    label_name = f'Rotulo_{name[:-4]}.txt'
    label = open(f'{path_rotulo}/{label_name}', "r")
    conteudo = label.readlines()
    for dado in conteudo:
        dado = dado.split("\t")
        start = float(dado[0])
        if(start < 0):
            start = 0
        end = float(dado[1])
        dado[2] = dado[2].replace("\n", "")
        rttm_train.write(
            f'SPEAKER {name[:-4]} 1 {start} {end-start} <NA> <NA> {dado[2]} <NA> <NA>\n')


# criando o RTTM para o validação

rttm_val = open("Cambio.development.rttm", "w")

for name in audios_val:
    label_name = f'Rotulo_{name[:-4]}.txt'
    label = open(f'{path_rotulo}/{label_name}', "r")
    conteudo = label.readlines()
    for dado in conteudo:
        dado = dado.split("\t")
        start = float(dado[0])
        if(start < 0):
            start = 0
        end = float(dado[1])
        dado[2] = dado[2].replace("\n", "")
        rttm_val.write(
            f'SPEAKER {name[:-4]} 1 {start} {end-start} <NA> <NA> {dado[2]} <NA> <NA>\n')


# criando o RTTM para o teste

rttm_test = open("Cambio.test.rttm", "w")

for name in audios_test:
    label_name = f'Rotulo_{name[:-4]}.txt'
    label = open(f'{path_rotulo}/{label_name}', "r")
    conteudo = label.readlines()
    for dado in conteudo:
        dado = dado.split("\t")
        start = float(dado[0])
        if(start < 0):
            start = 0
        end = float(dado[1])
        dado[2] = dado[2].replace("\n", "")
        rttm_test.write(
            f'SPEAKER {name[:-4]} 1 {start} {end-start} <NA> <NA> {dado[2]} <NA> <NA>\n')


# fechando os arquivos rttm

rttm_train.close()
rttm_test.close()
rttm_val.close()


# arquivos processados
print(f'processados : {qtd_treino+qtd_test+qtd_val}/{qtd_de_audio} arquivos')
print(f'treino : {qtd_treino} val: {qtd_val} test: {qtd_test}')
