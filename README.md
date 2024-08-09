# progettoDeepfakeAudio
Primo scheletro del progetto di rilevamento audio deepfake.\
Dataset utilizzato: FakeOrReal for-2sec https://bil.eecs.yorku.ca/datasets/ \
Modelli implementati: CNN a 1/2/3 layer convolutivi, ResNet18 leggermente modificato secondo fig 4e del paper https://arxiv.org/pdf/1603.05027v2 \
Cartella audio_data: circa 14k file audio, divisi a metà in fake e real, sottoposti a due giri di train_test_split per \
\
Cartella test_audio_data: 4000 file audio (2k e 2k) non utilizzati in alcun punto del codice, provenienti dallo stesso dataset. Divisi in fake e real, ne vengono prelevati 1000 ciascuno per fare predizioni
e comparare la performance dei modelli\
\
Cartella plots: grafici delle metriche dei vari modelli\
\
Cartella models: modelli keras salvati\
\
main.py Prima versione file principale del progetto. Necessita di lavoro di refactoring\
resnset.py Implementazione ResNet18\
\
Eseguito tuning su diversi iperparametri usando keras_tuner\
Prossimo passo: Non utilizzare più il json salvato localmente, è troppo pesante, impiega svariati minuti a caricare e può andare in memoryerror
