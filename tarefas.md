- Baixar novos datasets e anotações Check
- Mapear no arquivo database.yml Check
	+ Mover para diretórios Check
	+ Comparar voxsrc 2020/dev com 2021/dev (wav,rttm)
	+ E 2020/eval com 2021/test

- Carregar multiplos audios
	+ Dia Check
	+ Emb

- Passar pelo pipeline de diarização Check
- Calcular o DER total Check


04/08
- Dockerfile pyannote nvidia Check
	- Diarization development Check
	- Diarization test
	- Diarization workshop-test
	
- Salvar rttm workshop-test
- Consolidar rttms em um unico arquivo
- Submeter ao CodaLab

- Extrair e salvar os embeddings
	- HTF5 ou txt + zip
- Criar d-vectors a partir dos embeddings
- Alimentar uis-rnn com os d-vetors

- Configurar embeddings (window, step, projection)

- Adicionar etapas de VAD e Segmentation. Estudar uso de MFCCs