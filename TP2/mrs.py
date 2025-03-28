"""
Created on Tue Apr  6 13:03:06 2021

@author: rpp
"""

import librosa #https://librosa.org/    #sudo apt-get install -y ffmpeg (open mp3 files)
import librosa.display
import librosa.beat
# import sounddevice as sd  #https://anaconda.org/conda-forge/python-sounddevice
import warnings
import numpy as np
import matplotlib.pyplot as plt
import os

def normalizeFeatures(features):
    

def getStats(features):
    mean = np.mean(features) #media 
    stddv = np.std(features) #desvio padrao
    skew = scipy.stats.skew(features) #assimetria
    kurtosis = scipy.stats.kurtosis(features) #curtose
    median = np.median(features) #mediana
    max = np.max(features) #maximo
    min = np.min(features) #minimos

    return np.array([mean, stddv, skew, kurtosis, median, max, min])

def getFeatures(fName, sr, mono):
    Songs = [f for f in os.listdir(fName) if os.path.isfile(os.path.join(fName, f))]

    SongDuration = len(Songs)

    features = np.empty([SongDuration,190], dtype = object); 


    for song in Songs:
        song_path = os.path.join(fName, song)
        y, fs = librosa.load(song_path, sr = sr, mono = mono) # y é a taxa de amostragem, fs é a taxa de amostragem

        mfcc = librosa.feature.mfcc(y, sr = fs, n_mfcc = 13) #vamos usar as 13 primeiras bandas
        spec_centroid = librosa.feature.spectral_centroid(y, sr = fs) #centroide espectral
        spec_bandwidth = librosa.feature.spectral_bandwidth(y, sr = fs) #largura de banda espectral
        spec_contrast = librosa.feature.spectral_contrast(y, sr = fs) #contraste espectral
        spec_flatness = librosa.feature.spectral_flatness(y, sr = fs)
        spec_rolloff = librosa.feature.spectral_rolloff(y, sr = fs) #rolloff espectral
        f0 = librosa.yin(y, fs, fmin=20, fmax= fs/2) # 20 é o minimo que o ouvido humano consegue ouvir, fs/2 é a frequencia maxima que podemos obter (teorema de Nyquist)
        f0[f0 == fs/2] = 0 #substitui os valores que são iguais a fs/2 por 0 (o prof disse que quando tinhas a freq maxima para por a 0 por causa de erros na função)
        zero_cross_rate = librosa.feature.zero_crossing_rate(y) #quantas vezes a onda cruza o zero
        rms = librosa.feature.rms(y) #root mean square
        tempo = librosa.beat.tempo(y, sr = fs)

        for i in range(len(mfcc)):
            features[song, i * 7: i*7 + 7] = getStats(mfcc[i])
        
        features[song, 91:91 + 7] = getStats(spec_centroid[0])
        features[song, 98: 98 + 7] = getStats(spec_bandwidth[0])
        for i in range(len(spec_contrast)):
            features[song, 105 + i * 7: 105 + i * 7 + 7] = getStats(spec_contrast[i])
        
        features[song, 154: 154 + 7] = getStats(spec_flatness[0])
        features[song, 161: 161 + 7] = getStats(spec_rolloff[0])
        features[song, 168: 168 + 7] = getStats(f0)
        features[song, 175: 175 + 7] = getStats(zero_cross_rate[0])
        features[song, 182: 182 + 7] = getStats(rms[0])
        features[song, 189: 189 + 7] = getStats(tempo)
        






    return SongDuration


if __name__ == "__main__":
    plt.close('all')
    
    #--- Load file
    fName = os.path.join("./Queries/MT0000414517.mp3")   
    audiosPath = os.path.join("./Dataset/Audios") 
    sr = 22050
    mono = True
    warnings.filterwarnings("ignore")
    y, fs = librosa.load(fName, sr=sr, mono = mono)
    print(y.shape)
    print(fs)
    
    #--- Play Sound
    #sd.play(y, sr, blocking=False)
    
    #--- Plot sound waveform
    plt.figure()
    librosa.display.waveshow(y)
    
    #--- Plot spectrogram
    Y = np.abs(librosa.stft(y))
    Ydb = librosa.amplitude_to_db(Y, ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(Ydb, y_axis='linear', x_axis='time', ax=ax)
    ax.set_title('Power spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

    SongT = getFeatures(audiosPath, sr, mono)

        
    #--- Extract features    
    sc = librosa.feature.spectral_centroid(y = y)  #default parameters: sr = 22050 Hz, mono, window length = frame length = 92.88 ms e hop length = 23.22 ms 
    sc = sc[0, :]
    print(sc.shape)
    times = librosa.times_like(sc)
    plt.figure(), plt.plot(times, sc)
    plt.xlabel('Time (s)')
    plt.title('Spectral Centroid')
    