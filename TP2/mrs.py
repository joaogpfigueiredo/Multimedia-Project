"""
Created on Tue Apr  6 13:03:06 2021

@author: rpp
"""

import scipy
import librosa #https://librosa.org/    #sudo apt-get install -y ffmpeg (open mp3 files)
import librosa.display
from scipy.spatial.distance import cosine
# import sounddevice as sd  #https://anaconda.org/conda-forge/python-sounddevice
import warnings
import numpy as np
import matplotlib.pyplot as plt
import os
from ntpath import join
from genericpath import isfile
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

def read_features(input):
    return np.genfromtxt(input, delimiter=',')

def save_features(output, features):
    np.savetxt(output, features, fmt="%0.6f", delimiter=',')

def read_top_10(input):
    data = np.genfromtxt(input, delimiter=',', dtype=None, encoding=None)
    return [(name, float(dist)) for name, dist in data]

def read_Directory(path):
        return [f for f in os.listdir(path) if isfile(join(path, f))]

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def manhattan_distance(a, b):
    return np.sum(np.abs(a - b))

def cosine_distance(a, b):
    return cosine(a, b)

def normalizeFeatures(features):
    normalizedFeatures = np.zeros(features.shape)

    min_vals = np.min(features, axis=0) #minimo de cada coluna
    max_vals = np.max(features, axis=0) #maximo de cada coluna

    for i in range(features.shape[1]):
        max_val = np.max(features[:, i])
        min_val = np.min(features[:, i])

        if max_val == min_val:
            normalizedFeatures[:, i] = 0 #Se o max e min são iguais então a normalizacao é 0 para evitar erro matematico
        else:
            normalizedFeatures[:, i] = (features[:, i] - min_val) / (max_val - min_val) #Formula da normalizacao
        
    return normalizedFeatures, min_vals, max_vals

def implementedSC(signal, hopSize = 512, sr = 22050):

    windowSize = 2048
    N = len(signal)

    if N < windowSize:
        pad = windowSize - N
    else:
        pad = (hopSize - ((N - windowSize) % hopSize)) % hopSize

    signal_padded = np.concatenate([signal, np.zeros(pad)])

    windows = np.hanning(windowSize) 
    frequencies = np.fft.rfftfreq(windowSize, 1/sr)     

    spectrogram = []
    num_frames = (len(signal_padded) - windowSize) // hopSize + 1 
    
    for i in range(0, num_frames * hopSize, hopSize):
        frame = signal_padded[i:i+windowSize] * windows
        spectrum = np.abs(np.fft.rfft(frame))
        spectrogram.append(spectrum)
    
    centroids = []
    for frame in spectrogram:
        sum_magnitude = np.sum(frame)
        if sum_magnitude == 0:
            centroids.append(0)
        else:
            sc = np.sum(frequencies * frame) / sum_magnitude
            centroids.append(sc)
    
    return np.array(centroids)   

def calculate_and_compare(audios_folder, sr=22050):
     
    results = []

    for filename in os.listdir(audios_folder):
        file_path = os.path.join(audios_folder, filename)
        signal, sr = librosa.load(file_path, sr=sr, mono=True)
        my_centroid = implementedSC(signal)
        librosa_centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)

        sc = librosa_centroid[:, 2:]  
        my_centroid = my_centroid[:sc.shape[1]]  

        minSize = min(sc.shape[1], len(my_centroid))
        sc = sc[:, :minSize]
        my_centroid = my_centroid[:minSize]

        sc_flat = sc.flatten()

        correlation, _ = pearsonr(sc_flat, my_centroid)
        rmse = np.sqrt(mean_squared_error(sc_flat, my_centroid))

        results.append((correlation, rmse))
        
    print("Results saved successfully!")
    save_features("pearson_rmse_results.csv", results)

    return results


def similarity_measurements(query_features, features_list_normalized):

    if os.path.isfile("euclideanDistance.csv") and os.path.isfile("manhattanDistance.csv") and os.path.isfile("cosineDistance.csv"):
        euclideanDistance = read_features("euclideanDistance.csv")
        manhattanDistance = read_features("manhattanDistance.csv")
        cosineDistance = read_features("cosineDistance.csv")
        return euclideanDistance, manhattanDistance, cosineDistance

    num_songs = len(features_list_normalized) 

    euclideanDistance = np.zeros(num_songs)
    manhattanDistance = np.zeros(num_songs)
    cosineDistance = np.zeros(num_songs)

    for i in range(num_songs):
        euclideanDistance[i] = euclidean_distance(query_features, features_list_normalized[i])
        manhattanDistance[i] = manhattan_distance(query_features, features_list_normalized[i])
        cosineDistance[i] = cosine_distance(query_features, features_list_normalized[i])

    print("Distances calculated successfully!")

    save_features("euclideanDistance.csv", euclideanDistance)
    save_features("manhattanDistance.csv", manhattanDistance)
    save_features("cosineDistance.csv", cosineDistance)
    
    return euclideanDistance, manhattanDistance, cosineDistance


def create_similarity_rankings(euclideanDistance, manhattanDistance, cosineDistance, audio_folder_path):

    if os.path.isfile("ranking_euclidean.csv") and os.path.isfile("ranking_manhattan.csv") and os.path.isfile("ranking_cosine.csv"):
        top_10_euclidean = read_top_10("ranking_euclidean.csv")
        top_10_manhattan = read_top_10("ranking_manhattan.csv")
        top_10_cosine = read_top_10("ranking_cosine.csv")
        return top_10_euclidean, top_10_manhattan, top_10_cosine
    
    euclideanRanking = np.argsort(euclideanDistance)[:10]
    manhattanRanking = np.argsort(manhattanDistance)[:10] 
    cosineRanking = np.argsort(cosineDistance)[:10]

    listSongsNames = read_Directory(audio_folder_path)

    top_10_euclidean = [(listSongsNames[i], euclideanDistance[i]) for i in euclideanRanking]
    top_10_manhattan  = [(listSongsNames[i], manhattanDistance[i]) for i in manhattanRanking]
    top_10_cosine = [(listSongsNames[i], cosineDistance[i]) for i in cosineRanking]


    print("Top 10 Euclidean Distance Songs:")
    for name, dist in top_10_euclidean:
        print(f"{name}: {dist:.5f}")

    print("Top 10 Manhattan Distance Songs:")
    for name, dist in top_10_manhattan:
        print(f"{name}: {dist:.5f}")

    print("Top 10 Cosine Distance Songs:")
    for name, dist in top_10_cosine:
        print(f"{name}: {dist:.5f}")

    print("Rankings created successfully!")

    np.savetxt("ranking_euclidean.csv", top_10_euclidean, fmt="%s", delimiter=",")
    np.savetxt("ranking_manhattan.csv", top_10_manhattan, fmt="%s", delimiter=",")
    np.savetxt("ranking_cosine.csv", top_10_cosine, fmt="%s", delimiter=",")

    return top_10_euclidean, top_10_manhattan, top_10_cosine

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

    nSongs = len(Songs)

    features = np.empty([nSongs, 190], dtype = object); 


    for idx, song in enumerate(Songs):
        song_path = os.path.join(fName, song)
        y, fs = librosa.load(song_path, sr = sr, mono = mono) # y é a taxa de amostragem, fs é a taxa de amostragem

        mfcc = librosa.feature.mfcc(y=y, sr=fs, n_mfcc=13) #vamos usar as 13 primeiras bandas
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=fs) #centroide espectral
        spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=fs) #largura de banda espectral
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=fs) #contraste espectral
        spec_flatness = librosa.feature.spectral_flatness(y=y)
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=fs) #rolloff espectral
        f0 = librosa.yin(y=y, fmin=20, fmax=fs/2) # 20 é o minimo que o ouvido humano consegue ouvir, fs/2 é a frequencia maxima que podemos obter (teorema de Nyquist)
        f0[f0 == fs/2] = 0 #substitui os valores que são iguais a fs/2 por 0 (o prof disse que quando tinhas a freq maxima para por a 0 por causa de erros na função)
        zero_cross_rate = librosa.feature.zero_crossing_rate(y=y) #quantas vezes a onda cruza o zero
        rms = librosa.feature.rms(y=y) #root mean square
        tempo = librosa.feature.tempo(y=y, sr=fs)

        stats_mfcc = np.zeros((13, 7))
        stats_spec_contrast = np.zeros((7, 7))
        
        for i in range(len(stats_mfcc)):
            stats_mfcc[i] = getStats(mfcc[i])
        
        stats_mfcc = stats_mfcc.flatten()
        features[idx, 0:13 * 7] = stats_mfcc[0:13 * 7]
        
        features[idx, 91:91 + 7] = getStats(spec_centroid[0])
        features[idx, 98: 98 + 7] = getStats(spec_bandwidth[0])

        for i in range(len(stats_spec_contrast)):
            stats_spec_contrast[i] = getStats(spec_contrast[i])
        
        stats_spec_contrast = stats_spec_contrast.flatten()
        features[idx, 105:105 + 49] = stats_spec_contrast[0:49]
            
        
        features[idx, 154:154 + 7] = getStats(spec_flatness[0])
        features[idx, 161:161 + 7] = getStats(spec_rolloff[0])
        features[idx, 168:168 + 7] = getStats(f0)
        features[idx, 175:175 + 7] = getStats(zero_cross_rate[0])
        features[idx, 182:182 + 7] = getStats(rms[0])
        features[idx, 189:189 + 1] = tempo
        
    return features


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

    features = getFeatures(audiosPath, sr, mono)
    results = calculate_and_compare(audiosPath)
    euclideanDistance, manhattanDistance, cosineDistance = similarity_measurements(features[2], features)
    top_10_euclidean, top_10_manhattan, top_10_cosine = create_similarity_rankings(euclideanDistance, manhattanDistance, cosineDistance, audiosPath)
    
    # Ex 2.1.4
    normalized_features, min_vals, max_vals = normalizeFeatures(features)
    features_with_minmax = np.vstack([min_vals, max_vals, normalized_features])
    save_features("extracted_features.csv", features_with_minmax)
        
    #--- Extract features    
    sc = librosa.feature.spectral_centroid(y = y)  #default parameters: sr = 22050 Hz, mono, window length = frame length = 92.88 ms e hop length = 23.22 ms 
    sc = sc[0, :]
    print(sc.shape)
    times = librosa.times_like(sc)
    plt.figure(), plt.plot(times, sc)
    plt.xlabel('Time (s)')
    plt.title('Spectral Centroid')