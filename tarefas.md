12/08

- Criar d-vectors a partir dos embeddings
  - Carregar audios (protocol)
  - Entender estrutura files/folders dvector_create e portar para pyannote
  - Sample (Train + Test)
  - VoxConverse (Test + Dev)
  - VoxCeleb1 + VoxCeleb2
- Treinar uis-rnn com os d-vetors

04/08

- Dockerfile pyannote nvidia Check
  - Diarization development Check
  - Diarization test Check
  - Diarization workshop-test Check
- Salvar rttm workshop-test Check
- Consolidar rttms em um unico arquivo Check
- Submeter ao CodaLab Check

- Extrair e salvar os embeddings

  - HTF5 ou txt + zip

- Configurar embeddings (window, step, projection)

- Adicionar etapas de VAD e Segmentation. Estudar uso de MFCCs

Julho

- Baixar novos datasets e anotações Check
- Mapear no arquivo database.yml Check

  - Mover para diretórios Check
  - Comparar voxsrc 2020/dev com 2021/dev (wav,rttm)
  - E 2020/eval com 2021/test

- Carregar multiplos audios

  - Dia Check
  - Emb

- Passar pelo pipeline de diarização Check
- Calcular o DER total Check
