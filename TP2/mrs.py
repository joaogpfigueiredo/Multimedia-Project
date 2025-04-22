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

def save_normalized_features(output, features_list, min_vals, max_vals):


    data_to_save = [min_vals, max_vals]
    data_to_save.extend(features_list)

    np.savetxt(output, data_to_save, fmt="%0.5f", delimiter=',')

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

def normalized_features_query(a, b, features_list, min_vals, max_vals, output): 
    
    if os.path.isfile(output):
        return read_features(output)
    
    num_rows, num_cols = np.shape(features_list)

    for column in range(num_cols):
        fMin = min_vals[column]
        fMax = max_vals[column]

        if fMax == fMin:
            features_list[:, column] = 0
        else:
            features_list[:, column] = a + ((features_list[:, column] - fMin) * (b - a)) / (fMax - fMin)

    print("Features normalized successfully!")

    save_normalized_features(output, features_list, min_vals, max_vals)
    
    return features_list

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

def evaluate_centroid_folder(audios_folder, sr=22050):
     
    results = []

    for filename in os.listdir(audios_folder):
        file_path = os.path.join(audios_folder, filename)
        signal, sr = librosa.load(file_path, sr=sr, mono=True)
        my_centroid = implementedSC(signal)
        librosa_centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)

        # Adjust the starting index as necessary to match the data
        sc = librosa_centroid[:, 2:]  
        my_centroid = my_centroid[:sc.shape[1]]  

        # Ensure that the two arrays have the same length
        minSize = min(sc.shape[1], len(my_centroid))
        sc = sc[:, :minSize]
        my_centroid = my_centroid[:minSize]


        sc_flat = sc.flatten() # Flatten the 2D array to 1D

        correlation, _ = pearsonr(sc_flat, my_centroid)
        rmse = np.sqrt(mean_squared_error(sc_flat, my_centroid))

        results.append((correlation, rmse))
        
    print("Results saved successfully!")
    save_features("pearson_rmse_results.csv", results)

    return results


def compute_distances(query_vec, features_norm):
    """Return Euclidean, Manhattan and Cosine distances to every song.

    Uses cached CSVs if they already exist.
    """
    if (all(os.path.isfile(f)
            for f in ("euclideanDistance.csv",
                      "manhattanDistance.csv",
                      "cosineDistance.csv"))):
        return (read_features("euclideanDistance.csv"),
                read_features("manhattanDistance.csv"),
                read_features("cosineDistance.csv"))

    n = len(features_norm)
    euclid   = np.zeros(n)
    manhat   = np.zeros(n)
    cosine   = np.zeros(n)

    for i, feat in enumerate(features_norm):
        euclid[i] = euclidean_distance(query_vec, feat)
        manhat[i] = manhattan_distance(query_vec, feat)
        cosine[i] = cosine_distance(query_vec, feat)

    print("Distances computed successfully!")
    save_features("euclideanDistance.csv", euclid)
    save_features("manhattanDistance.csv", manhat)
    save_features("cosineDistance.csv", cosine)

    return euclid, manhat, cosine

def build_top10_rankings(euclid, manhat, cosine, audio_folder):
    """Return three Top‑10 lists (Euclidean, Manhattan, Cosine)."""
    if (all(os.path.isfile(f)
            for f in ("ranking_euclidean.csv",
                      "ranking_manhattan.csv",
                      "ranking_cosine.csv"))):
        return (read_top_10("ranking_euclidean.csv"),
                read_top_10("ranking_manhattan.csv"),
                read_top_10("ranking_cosine.csv"))

    song_names = read_Directory(audio_folder)

    idx_euclid = np.argsort(euclid)[:11]
    idx_manhat = np.argsort(manhat)[:11]
    idx_cosine = np.argsort(cosine)[:11]

    top10_euclid  = [(song_names[i], euclid[i]) for i in idx_euclid[1:]]
    top10_manhat  = [(song_names[i], manhat[i]) for i in idx_manhat[1:]]
    top10_cosine  = [(song_names[i], cosine[i]) for i in idx_cosine[1:]]

    # print nicely
    def _print_list(title, lst):
        print(f"Top 10 {title}:")
        for name, dist in lst:
            print(f"{name}: {dist:.5f}")

    _print_list("Euclidean Distance", top10_euclid)
    _print_list("Manhattan Distance", top10_manhat)
    _print_list("Cosine Distance",    top10_cosine)

    # save CSVs
    np.savetxt("ranking_euclidean.csv",  top10_euclid,  fmt="%s", delimiter=",")
    np.savetxt("ranking_manhattan.csv",  top10_manhat,  fmt="%s", delimiter=",")
    np.savetxt("ranking_cosine.csv",     top10_cosine,  fmt="%s", delimiter=",")

    print("Rankings created successfully!")
    return top10_euclid, top10_manhat, top10_cosine

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
    fName = os.path.join("./Queries", "MT0000414517.mp3")   
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

    features_all = getFeatures(audiosPath, sr, mono)

    normalized_features, min_vals, max_vals = normalizeFeatures(features_all)
    features_with_minmax = np.vstack([min_vals, max_vals, normalized_features])
    save_features("extracted_features.csv", features_with_minmax)

    features_query = getFeatures('./Queries/', sr, mono)
    features_normalized_query = normalized_features_query(0,1,features_query, min_vals, max_vals, "featuresNormalizedQuery.csv")
        
    results = evaluate_centroid_folder(audiosPath)
    euclideanDistance, manhattanDistance, cosineDistance = compute_distances(features_normalized_query[2], normalized_features)
    top_10_euclidean, top_10_manhattan, top_10_cosine = build_top10_rankings(euclideanDistance, manhattanDistance, cosineDistance, audiosPath)

    #--- Extract features    
    sc = librosa.feature.spectral_centroid(y = y)  #default parameters: sr = 22050 Hz, mono, window length = frame length = 92.88 ms e hop length = 23.22 ms 
    sc = sc[0, :]
    print(sc.shape)
    times = librosa.times_like(sc)
    plt.figure(), plt.plot(times, sc)
    plt.xlabel('Time (s)')
    plt.title('Spectral Centroid')