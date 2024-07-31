Primo scheletro del progetto di rilevamento audio deepfake.\
Cartella audio_data: circa 11k file audio, divisi in fake e real. Tutti utilizzati per popolare il json dei dati. Usati anche per l'apprendimento, divisi in training, validation e test.\
\
Cartella test_audio_data: 1100 file audio non utilizzati in alcun punto del codice, provenienti dallo stesso dataset. Divisi in fake e real, al momento ne vengono prelevati 300 per classe per fare predizioni
e comparare la performance dei modelli\
\
Cartella plots: grafici delle metriche dei vari modelli\
\
Cartella models: modelli keras salvati\
\
main.py primo abbozzo di progetto, ancora nessun effort di refactoring o simili. Per ora un solo modello addestrato, una rete convoluzionale\
\
Possibili passi successivi: 
Fase preliminare di data augmentation e/o grafici di esempio delle feature usate (mfcc / mel spectrogram);\
Creazione e training di modelli diversi ispirandosi ai vari paper sull'argomento (LCNN/LSTM/RNN/GMM/RawNet);\
Ricerca di possibili sintomi di overfitting/underfitting;\
Tuning degli iperparametri quali learning rate/numero di epoche di apprendimento;\
