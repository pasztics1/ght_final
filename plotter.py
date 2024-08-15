#cd C:\Users\Surface\Desktop\zeto\eeg_testing_plugin
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt, welch, iirnotch
file_name = 'ZE-030-025-056-raw.txt'

with open(file_name, 'r') as file:
    lines = file.readlines()

data_lines = lines[2:] #first line capture id, second line column names
cleaned_data = []


i=0
for line in data_lines:
    values = line.split()
    values = [float(val) for val in values]
    cleaned_data.append(values)
    i+=1

data_array = np.array(cleaned_data)

column_names = lines[1].split()[1:20]
print(column_names)
column_names = [column_names[i].split('[')[0] for i in range(len(column_names))] #stripping the metric units from the names (since they are redundant),
                                                                                 #the times are in seconds, and the signal is mesured in microvolts-

data = [[] for _ in range(len(column_names))]
for i in range(1,len(column_names)+1): #time is the first coulumn, which we don't need in this case. the time can be calculated if needed, since sr and index is known
    for j in range(len(data_array)):
        data[i-1].append(data_array[j,i])
        
signal=[[] for _ in range(len(column_names))]
sampling_rate = 500 #/sec
periods = 5 #in seconds
lenght = (sampling_rate*periods)
total_sampling = len(data_array)

for i in range((total_sampling//(lenght))-1):
        signal[i%len(column_names)].append(data[i%len(column_names)][i*lenght:(i+1)*(sampling_rate*periods)])

signal[(total_sampling//(lenght)%len(column_names))-1].append(data[(len(data)//(lenght))%len(column_names)][((total_sampling//(lenght))*lenght):total_sampling])


for i in range(2500-len(signal[(total_sampling//(lenght)%len(column_names))-1][len(signal[(total_sampling//(lenght)%len(column_names))])])): #in order for the last 5s to be exactly 5*sr long. the remainder's filled with 0s.
    signal[(total_sampling//(lenght)%len(column_names))-1][len(signal[(total_sampling//(lenght)%len(column_names))])].append(0)
    
print((total_sampling//(lenght)%len(column_names))-1,len(signal[(total_sampling//(lenght)%len(column_names))-1][1]))
#"signal" is a 2d list with 19 sublists, containing only the data we are interested in performing the fourier transformation on.
#the "data" list contains a 2d list, with len(column_names) list in it (19 channels optimally for the 19 electrodes)



def apply_filters(signal, sampling_rate, lff=1.5, hff=60.0, notch_freq=50.0, notch_quality=30.0, pad_len=100):
    nyquist = 0.5 * sampling_rate
    
    # High-pass filter (Low-Frequency Filter)
    low = lff / nyquist
    b, a = butter(2, low, btype='high')  # Increased order to 2 for better attenuation
    padded_signal = np.pad(signal, pad_len, mode='edge')
    filtered_signal = filtfilt(b, a, padded_signal)
    filtered_signal = filtered_signal[pad_len:-pad_len]
    
    # Low-pass filter (High-Frequency Filter)
    high = hff / nyquist
    b, a = butter(2, high, btype='low')  # Increased order to 2 for better attenuation
    padded_signal = np.pad(filtered_signal, pad_len, mode='edge')
    filtered_signal = filtfilt(b, a, padded_signal)
    filtered_signal = filtered_signal[pad_len:-pad_len]

    # Bandpass filter
    b, a = butter(2, [low, high], btype='band')  # Bandpass filter
    padded_signal = np.pad(filtered_signal, pad_len, mode='edge')
    filtered_signal = filtfilt(b, a, padded_signal)
    filtered_signal = filtered_signal[pad_len:-pad_len]

    # Notch filter
    notch_freq = notch_freq / nyquist
    b, a = iirnotch(notch_freq, notch_quality)
    padded_signal = np.pad(filtered_signal, pad_len, mode='edge')
    filtered_signal = filtfilt(b, a, padded_signal)
    filtered_signal = filtered_signal[pad_len:-pad_len]

    return filtered_signal




def analyser(original_signal, sampling_rate, target_freq, plotting=False):
    filtered_signal = apply_filters(original_signal, sampling_rate)

    # Compute the FFT of the filtered signal
    if len(filtered_signal) % 2:
        signal_fft = np.fft.fft(filtered_signal)[0:(len(filtered_signal) + 1) // 2]
    else:
        signal_fft = np.fft.fft(filtered_signal)[0:int(1 + (len(filtered_signal) / 2))]

    freqs = np.fft.fftfreq(len(filtered_signal), 1.0 / sampling_rate)

    # Selecting the positive frequencies
    positive_freq_indices = np.where(freqs >= 0)
    positive_freqs = freqs[positive_freq_indices]
    fft_positive = signal_fft[positive_freq_indices]

    magnitudes = np.abs(fft_positive) / len(signal_fft)
    phases = np.angle(fft_positive) + (np.pi / 2)  # to normalize the lag compared to cos(x)
    phases = (phases + np.pi) % (2 * np.pi) - np.pi  # getting the phases into the range of [-pi, pi]

    # Power Spectral Density (PSD)
    f_psd, Pxx_den = welch(filtered_signal, sampling_rate, nperseg=1024)

    # Ensure Pxx_den is a 1-D array
    Pxx_den = np.squeeze(Pxx_den)

    # Finding the peaks in the PSD
    threshold = np.max(Pxx_den) * 0.5  #!!
    peaks, _ = find_peaks(Pxx_den, height=threshold)

    significant_freqs = f_psd[peaks]
    significant_powers = Pxx_den[peaks]

    if plotting:
        # Plotting the filtered signal
        plt.figure(figsize=(12, 6))

        plt.subplot(3, 1, 1)
        plt.plot(np.linspace(0, len(original_signal)/sampling_rate, len(original_signal)), original_signal)
        plt.title('Original Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

        plt.subplot(3, 1, 2)
        plt.plot(np.linspace(0, len(filtered_signal)/sampling_rate, len(filtered_signal)), filtered_signal)
        plt.title('Filtered Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

        # Magnitude Spectrum (Magnitude in terms of frequency)
        plt.subplot(3, 1, 3)
        plt.semilogy(f_psd, Pxx_den)
        plt.semilogy(significant_freqs, significant_powers, 'x')
        plt.title('Power Spectral Density')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')

        plt.tight_layout()
        plt.show()

    return {
        'frequencies': significant_freqs,
        'powers': significant_powers
    }





x = np.linspace(0,100,len(signal[0][0][1000:1500]))
plt.plot(x, apply_filters(signal[0][0],sampling_rate)[1000:1500])
# Show every other x-axis value
plt.xticks(np.arange(min(x), max(x) + 1))
plt.ylabel("Y-axis")
plt.xlabel("X-axis")
plt.grid(True)
plt.show()

detected = []
for i in range(5):
    detected.append(analyser(apply_filters(signal[0][0][500*i:500*(i+1)],500),500,4))

for i in range(10):
    print(analyser(apply_filters(signal[i][0][1000:2000],500),500,4))



'''
print(signal[0][0])
electrodes = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
frequencies_found = []
for i in range(len(signal)):
    frequencies_found.append([])
    print(f"\nFound frequencies for {electrodes[i]}: ")
    for j in range(len(signal[i])):
        frequencies_found[i].append(detect_sinusoids(signal[i][j]))
        print(f"{detect_sinusoids(signal[i][j])} ")
    
        
        
# Example usage
# Assuming you have a 2D list 'signals' with 19 lists of measured frequencies and a sampling rate of 'sampling_rate'
# signals = [[...], [...], ..., [...]]  # 19 channels of signals
# sampling_rate = 1000  # Example sampling rate in Hz

# detected_frequencies = detect_sinusoids(signals, sampling_rate)
# print(detected_frequencies)
'''




import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

TESTING = False


def analyse(original_signal, sampling_rate,plotting=True):
    if len(original_signal)%2:
        signal_fft = np.fft.fft(original_signal)[0:int((len(original_signal)+1)/2)]
    else:
        signal_fft = np.fft.fft(original_signal)[0:int(1+(len(original_signal)/2))]

    freqs = np.fft.fftfreq(len(original_signal), 1.0/sampling_rate)

    # Selecting the positive frequencies
    positive_freq_indices = np.where(freqs >= 0)
    positive_freqs = freqs[positive_freq_indices]
    fft_positive = signal_fft[positive_freq_indices]

    magnitudes = np.abs(fft_positive)/len(signal_fft)
    phases = np.angle(fft_positive)+(np.pi/2) #to normalise the lag compared to cos(x)
    phases = (phases + np.pi) % (2 * np.pi) - np.pi #getting the phases into the range of [-pi,pi]


    #Finding the peaks
    threshold = 0.1
    peaks, _ = find_peaks(magnitudes, height=threshold)

    # Extracting significant frequencies, magnitudes, and phases
    global significant_freqs, significant_magnitudes, significant_phases
    significant_freqs = positive_freqs[peaks]
    significant_magnitudes = magnitudes[peaks]

    significant_phases = phases[peaks]

    if plotting:
        # Plotting (Magnitude in terms of time)
        plt.figure(figsize=(12, 6))

        # Magnitude Spectrum (Magnitude in terms of frequency)
        plt.subplot(2, 1, 1)
        plt.stem(significant_freqs, significant_magnitudes)
        plt.title('Magnitude Spectrum')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')

        # Phase Spectrum (Phase in terms of frequency)
        plt.subplot(2, 1, 2)
        plt.stem(significant_freqs, significant_phases)
        plt.title('Phase Spectrum')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Phase (radians)')

        plt.tight_layout()
        plt.show()
    return significant_freqs

