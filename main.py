import logging
from typing import Optional
from reportlab.pdfgen import canvas
from zcp_py_plugin.meta import Meta
from zcp_py_plugin.phi import Phi
from zcp_py_plugin.report_plugin import ReportPlugin
from zcp_py_plugin.plugin_result_type import PluginResultType
from zcp_py_plugin.plugin_client import PluginClient
import mne
import shutil

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt, welch, iirnotch, lfilter
import statistics
from reportlab.pdfgen import canvas
import textwrap



def apply_filters(signal, sampling_rate, lff=1.0, hff=15.0, notch_freq=50.0, notch_quality=5.0, pad_len=500):  # Increased pad_len
    nyquist = 0.5 * sampling_rate
    
    # Padding with a different mode, e.g., 'symmetric'
    padded_signal = np.pad(signal, pad_len, mode='symmetric')
    
    # High-pass filter (Low-Frequency Filter)
    low = lff / nyquist
    b, a = butter(2, low, btype='high')
    filtered_signal = filtfilt(b, a, padded_signal)
    
    # Low-pass filter (High-Frequency Filter)
    high = hff / nyquist
    b, a = butter(2, high, btype='low')
    filtered_signal = filtfilt(b, a, filtered_signal)
    
    # Notch filter
    notch_freq = notch_freq / nyquist
    b, a = iirnotch(notch_freq, notch_quality)
    filtered_signal = filtfilt(b, a, filtered_signal)
    
    # Remove the padding
    filtered_signal = filtered_signal[pad_len:-pad_len]
    
    return filtered_signal


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def detect_main_frequency(signal, sampling_rate, lowcut=None, highcut=None, min_magnitude=0.03, plotting=False):
    """
    Detects the main frequency in a given signal, based on a minimum magnitude threshold.
    
    Parameters:
    - signal: The input signal to analyze.
    - sampling_rate: The sampling rate of the signal.
    - lowcut: The low cutoff frequency for bandpass filtering (optional).
    - highcut: The high cutoff frequency for bandpass filtering (optional).
    - min_magnitude: Minimum magnitude value (in volts) to consider a frequency significant.
    - plotting: Boolean to indicate whether to plot the FFT magnitude spectrum.
    
    Returns:
    - main_freq: The detected main frequency, or None if no significant frequency is found.
    """
    
    # Apply Filters if specified (assuming bandpass_filter is defined elsewhere)
    if lowcut and highcut:
        filtered_signal = bandpass_filter(signal, lowcut, highcut, sampling_rate)
    else:
        filtered_signal = signal  # No filtering if not specified
    
    n = len(filtered_signal)
    signal_fft = np.fft.fft(filtered_signal)
    freqs = np.fft.fftfreq(n, 1.0 / sampling_rate)
    
    # Consider only the positive frequencies
    positive_freq_indices = np.where(freqs >= 0)
    positive_freqs = freqs[positive_freq_indices]
    fft_magnitudes = np.abs(signal_fft[positive_freq_indices])

    # Identifying the Main Frequency
    # Apply a minimum magnitude threshold
    significant_indices = np.where(fft_magnitudes >= min_magnitude)[0]

    if len(significant_indices) == 0:
        main_freq = None  # No significant frequency found
    else:
        # Find the peak with the maximum magnitude among significant indices
        max_magnitude_index = np.argmax(fft_magnitudes[significant_indices])
        main_freq = positive_freqs[significant_indices[max_magnitude_index]]

    # Plotting (if requested)
    if plotting:
        plt.figure(figsize=(12, 8))

        plt.subplot(3, 1, 1)
        plt.plot(np.linspace(0, len(signal) / sampling_rate, len(signal)), signal)
        plt.title('Original Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

        plt.subplot(3, 1, 2)
        plt.plot(np.linspace(0, len(filtered_signal) / sampling_rate, len(filtered_signal)), filtered_signal)
        plt.title('Filtered Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

        plt.subplot(3, 1, 3)
        plt.plot(positive_freqs, fft_magnitudes)
        if main_freq is not None:
            plt.plot(main_freq, fft_magnitudes[np.where(positive_freqs == main_freq)][0], 'rX')
        plt.title('FFT Magnitude Spectrum')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.axhline(y=min_magnitude, color='r', linestyle='--', label=f'Min Magnitude ({min_magnitude})')
        plt.legend()

        plt.tight_layout()
        plt.show()

    return main_freq






def analyser(original_signal, sampling_rate, plotting=False):
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
    

    


def generate_pdf_report(file_name, result):
    report_file_name = f'{file_name}_report.pdf'
    c = canvas.Canvas(report_file_name)
    electrodes = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
    
    # Set initial Y position
    y = 750
    line_height = 20
    bottom_margin = 50
    max_line_width = 400  # Maximum width for the text
    
    # Set font size
    c.setFont("Helvetica", 10)
    
    # Title
    c.drawString(100, y, f'GHT testing results for {file_name}')
    y -= line_height * 2
    
    # Measured signals
    for i, freq in enumerate(result):
        text = f"{electrodes[i]} : Frequency(es): {freq}"
        
        # Wrap text
        wrapped_text = textwrap.wrap(text, width=int(max_line_width / 6))  # Adjust wrap width to fit page
        
        for line in wrapped_text:
            c.drawString(100, y, line)
            y -= line_height
            
            # Check if we need a new page
            if y < bottom_margin:
                c.showPage()
                y = 750
    
    c.save()
    print(f"Report saved as: {report_file_name}")
    return report_file_name

    
    
class MyFirstPlugin(ReportPlugin):
    #ZE-030-025-056
    def process(self, meta: Meta, phi: Optional[Phi]) -> None:
        
        file_name = self.get_edf_file()
        new_file_name = 'C:\\Users\\Surface\\Desktop\\zeto\\zeto_eeg_testing_plugin_final\\random.edf' #az edf file dekodolt
        shutil.copyfile(file_name, new_file_name)

        data = mne.io.read_raw_edf(new_file_name)
        record = data.get_data(picks=data.ch_names[:19])
        info = data.info
        column_names = data.ch_names[:19]
        sampling_rate = meta.sample_rate #/sec (250 or 500)
        electrodes = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
        print(f"Running on a {sampling_rate}hz sampling rate")
        
        #PARAMETERS#
        periods = 5 #in seconds
        searched_frequency = 4 #hz
        margin = searched_frequency * 0.05
        median = True #ilyen formában nem fog működni a szignifikáns frekvencia, mert óaz mindig 50hz lesz a zaj miatt (maximum szűrővel, lehet úgy jobban is működik)
        plotting = True #nem tudom van e rá szükség
        #PARAMETERS#

        print(info)
        data=record
                
        signal=[[] for _ in range(len(column_names))]
        lenght = (sampling_rate*periods)
        total_sampling = meta.sample_count
        number_of_cycles = 1+total_sampling//lenght
            
        #test for shifts in reading the data
        """
        for i in range(19):
            x = np.linspace(0,1000,len(data[i]))
            plt.plot(x, apply_filters(data[i],sampling_rate))
            # Show every other x-axis value
            plt.xticks(np.arange(min(x), max(x) + 1))
            plt.ylabel("Y-axis")
            plt.xlabel("X-axis")
            plt.grid(True)
            plt.show()
        """
        #
        
        

        '''
        if plotting:
            for i in range(5): #nyilván out of ranget ad, ha nem elég hosszú a felvétel, legalább x*periods másodperc hosszú kell legyen
                x = np.linspace(0,100,len(signal[i][0]))
                plt.plot(x, apply_filters(signal[i][0],sampling_rate))
                # Show every other x-axis value
                plt.xticks(np.arange(min(x), max(x) + 1))
                plt.ylabel("Y-axis")
                plt.xlabel("X-axis")
                plt.grid(True)
                plt.show()
        '''
        
        #finding every frequency
        detected = []
        detected = [[] for _ in range(len(column_names))]
        for i in range(len(column_names)):
            for j in range(number_of_cycles-1): #accounting for the indexing 
                temp = detect_main_frequency(apply_filters(data[i][j*lenght:(j+1)*lenght], sampling_rate),sampling_rate) #['frequencies']
                detected[i].append(temp)
                #detected[i].append(statistics.mode(temp)) #if using the mode method
                
        #evaluation        
        result = [] #result is a list that contains whether or not a period was correct or not
        result = [[] for _ in range(len(column_names))]
        incorrect = [] #incorrect hold the frequencies that were picked up wrongfully. It contains 19 sub arrays that contains 2 element lists of the errors, with their index (human readable), and the detected frequency
        incorrect = [[] for _ in range(len(column_names))]
        
        
        for i in range(len(column_names)):
            correct = True
            other_signal = False
            for j in range(len(detected[i])): #this is a fix number, this should not fluctuate, and will not since it's lenght is number_of_cycles-1    
                if j%len(column_names) == i:
                    #checking the signal where it was given to the electrode
                    if detected[i][j] == searched_frequency:
                        pass
                    else:
                        correct=False 
                else:
                    if detected[i][j] != None:
                        other_signal = True
                        
                        incorrect[i].append([(j+1), detected[i][j]])
                    #checking whether or not there are any extra signals
            #evaluation
            if correct:
                if other_signal:
                    result[i].append(f"The {electrodes[i]} picked up the sine correctly, but other signals were detected; {incorrect[i]}.")
                else:
                    result[i].append(f"The {electrodes[i]} is working correctly, and no other signals were detected.")
                    
            else:
                if other_signal:
                    result[i].append(f"The {electrodes[i]} did not pick up the sine, and there were also other signals detected; {incorrect[i]}.")
                else:
                    result[i].append(f"The {electrodes[i]} did not pick up the sine, and there were no other signals detected.")
                    
        #plotting
            
        fig, axs = plt.subplots(4, 5, figsize=(20, 16))  # Adjust figsize as needed

        # Flatten the array of axes for easier indexing
        axs = axs.flatten()

        # Plot only when condition is met
        plot_index = 0  # Index to keep track of which subplot to use
        for i in range(len(column_names)):
            for j in range(number_of_cycles-1):  # Adjusted for fixed number of plots
                if j % len(column_names) == i:
                    x = np.linspace(0, 100, lenght)  # Create x-axis values
                    segment = apply_filters(data[i][j*lenght:(j+1)*lenght], sampling_rate)
                    axs[plot_index].plot(x, segment)
                    axs[plot_index].set_ylabel("Y-axis")
                    axs[plot_index].set_xlabel("X-axis")
                    axs[plot_index].grid(True)
                    axs[plot_index].set_xticks(np.arange(min(x), max(x) + 1, step=10))  # Adjust step as needed
                    plot_index += 1

        # Turn off any unused subplots
        for k in range(plot_index, len(axs)):
            axs[k].axis('off')

        # Adjust layout to prevent overlapping
        plt.tight_layout()
        plt.show()
            
    
            
                
        """old implementation
        for i in range(len(column_names)):
            for j in range(len(signal[i])):
                for k in range(periods):
                    if median:
                        temp = analyser(apply_filters(signal[i][j][sampling_rate*k:sampling_rate*(k+1)],sampling_rate),sampling_rate) # yet to be done
                        detected[i].append(temp['frequencies'])
                    else:
                        detected[i].append(detect_main_frequency(apply_filters(signal[i][j][sampling_rate*k:sampling_rate*(k+1)],sampling_rate), sampling_rate))
        
        
        
        
        #Janostol megkerdezni hogy hogyan, milyen szigorúan működjön az algoritmus? Első, utolsó másodpercet érdemes -e discardolni?
        #result contains the processed results, it is supposed to be a 2d array with 19 sub-arrays
        result = []
        result = [[] for _ in range(len(column_names))]
        if median:
            for i in range(len(detected)):
                if len(detected[i][0])>0: #[i][0] since we are only checking for the first 19 channels
                    if statistics.mode(detected[i][0]) == searched_frequency:
                        result[i].append(f'The searched frequency was found for {i+1}')
                    else:
                        result[i].append(f'The searched frequency was not found for channel: {i+1}, it was {detected[i][0]}')
        else:
            for i in range(len(detected)):
                    if detected[i][0] == searched_frequency:
                        result[i].append(f'The searched frequency was found for {i+1}')
                    else:
                        result[i].append(f'The searched frequency was not found for channel: {i+1}, it was {detected[i][0]}')
            
        """



        pdf_file_name= file_name.split("\\")[-1]
        report_file_name = generate_pdf_report(pdf_file_name,result)
        
        self.progress(80, 'Uploading report')
        self.emit(PluginResultType.PDF, file_name=report_file_name)
        self.progress(100, 'All done')
        logging.info(f"Emit succeeded")

def main():
    logging.basicConfig(level=logging.INFO)
    
        #for hosting plugin on the prod2 server
    PluginClient.of()\
        .server('apidev.zetoserver.com')\
        .vendor('Zeto')\
        .name('Automated GHT testing plugin x')\
        .version('1.0.0')\
        .operation_pdf()\
        .plugin(MyFirstPlugin)\
        .auth_string("eyJjZXJ0aWZpY2F0ZSI6ICItLS0tLUJFR0lOIENFUlRJRklDQVRFLS0tLS1cbk1JSURXakNDQWtLZ0F3SUJBZ0lSQVAzYjEyZkhVdTUvT1BGY3FNUmlCVDB3RFFZSktvWklodmNOQVFFTEJRQXdcbkZqRVVNQklHQTFVRUF3d0xSV0Z6ZVMxU1UwRWdRMEV3SGhjTk1qTXhNakF4TURneE1UVXlXaGNOTWpZd016QTFcbk1EZ3hNVFV5V2pBaE1SOHdIUVlEVlFRRERCWndiSFZuYVc1QWVtVjBieXhhWlhSdkxDeFVjblZsTUlJQklqQU5cbkJna3Foa2lHOXcwQkFRRUZBQU9DQVE4QU1JSUJDZ0tDQVFFQXhNTXdPWEZ4SGZ5alZwNVNsSjh5VUhFczV4ZDVcbmx2NDNPWHMxRGU1KytNbGQzT1AwOW1TVXUyVmNGTXJraG42UHArbkFpcUpBS0I0UXp5aS8ya2lSTlpsbDI0SHFcblNOWk9kdlFoTi85MGdFTi9BbkppbGJrS2VPdUlCSkZ6UHdhOCtCL3I1Mk5kc1NwdFU4ZXFQK3JEY3c5TTZXSnJcbisyOFl6WTJJd1hEL1J6NTMwZjlvVDg1aDhUMitKN2NWcWwzWW5EYnR0dzZuYTVHVk0wS1RxbXJseEgvZ1BuNkRcbjJwak90SUhqNVYwa2dLci9lVWhiZVNmaW5YZFlwd2w1QmxpTGR2b2hvSHBVR1hjczVVNmpPODJpckE1NzV6NStcbjVqT2RiN1VFdkVFUW1sSEI2TnJONUlmMUVXbHA1SkdXVlFhTWZkT1FseWljM1YxeXpmRjBvbVRMdFFJREFRQUJcbm80R1hNSUdVTUFrR0ExVWRFd1FDTUFBd0hRWURWUjBPQkJZRUZHK0tuQmFBdTZjVXlCRTMxZ1ZIeVRBZThLNDJcbk1FWUdBMVVkSXdRL01EMkFGS1pVSlpSSmg2RmpMV1ZaZXNYd3B6cTFkb3BLb1Jxa0dEQVdNUlF3RWdZRFZRUURcbkRBdEZZWE41TFZKVFFTQkRRWUlKQUxFYTRyajNsSVpYTUJNR0ExVWRKUVFNTUFvR0NDc0dBUVVGQndNQ01Bc0dcbkExVWREd1FFQXdJSGdEQU5CZ2txaGtpRzl3MEJBUXNGQUFPQ0FRRUFyTWRkakNrVitTR0pNUTNLWlpNeEdMMllcbnVtSU5NYTM0ZnNJNzdDQTBTY2ZCb2htaVIzaTdMa2wwVWJ4c3RNUENIaWJLUlpySno3QXZJa202Z2l1dm1QcE1cbjY4QlIwVGFnU2Ywdk9PM2NCbG1qMHR1WEo3WDRYN2VBS0ZCL2huWkRZWld2QWdlZHVydWs4SmEwWXkyVklHZUxcblFOVWVSazRsVTMwNTZpTzhlYUVFNFhmZXRqMmROcXd2VXBHMzNvL1NlNnFDaWRvdy8veENSbnkwTlpBdHl2Z3Vcbkw5d1gzY3kycU1KYldoWGttRlBTZjJVdktpZmNMMkpZdG9Pa0gvR0sxS2lZU3RPenhMNkRuQ1VQMWd5UWhCbExcbkhhbkZSb3lvYklydUV0Qms2SU9IMFcvVU9SWXMwT0RJRmdRRU1rTXozMWFZNXZucHdMV2xkdkdDTGxjVy93PT1cbi0tLS0tRU5EIENFUlRJRklDQVRFLS0tLS0iLCAicHJpdmtleSI6ICItLS0tLUJFR0lOIFBSSVZBVEUgS0VZLS0tLS1cbk1JSUV2QUlCQURBTkJna3Foa2lHOXcwQkFRRUZBQVNDQktZd2dnU2lBZ0VBQW9JQkFRREV3ekE1Y1hFZC9LTldcbm5sS1VuekpRY1N6bkYzbVcvamM1ZXpVTjduNzR5VjNjNC9UMlpKUzdaVndVeXVTR2ZvK242Y0NLb2tBb0hoRFBcbktML2FTSkUxbVdYYmdlcEkxazUyOUNFMy8zU0FRMzhDY21LVnVRcDQ2NGdFa1hNL0JyejRIK3ZuWTEyeEttMVRcbng2by82c056RDB6cFltdjdieGpOallqQmNQOUhQbmZSLzJoUHptSHhQYjRudHhXcVhkaWNOdTIzRHFkcmtaVXpcblFwT3FhdVhFZitBK2ZvUGFtTTYwZ2VQbFhTU0Fxdjk1U0Z0NUorS2RkMWluQ1hrR1dJdDIraUdnZWxRWmR5emxcblRxTTd6YUtzRG52blBuN21NNTF2dFFTOFFSQ2FVY0hvMnMza2gvVVJhV25ra1paVkJveDkwNUNYS0p6ZFhYTE5cbjhYU2laTXUxQWdNQkFBRUNnZ0VBS2ZWenViUXl5cEcrNVRCQzdQV2IzYUtjMERUbDFXaWxyeWpTY2dPYmFRTkxcbjlGaGFPeGJNenI5NUtPZnhYcXJyaUlPazd3dFZnaGlUUGhIekE2SDQ4VVNpZjNKUFd6UDBMSkszNk1DZGYrS2tcbjJZazU4N0t2aElTNWp6dlRKeXdSTFJwbGpJVFlqSGkvTXAvLzhyeGw3SW5sUFZtakxFMlBMUHBUSU9rdlR1b25cbngyMWNyNnZiZVpDcWR3S1Z4bTgyTHB1Y09YajFOcUMySUNYY1kwK2ZJc1hhNUZJL1VoNzlBVW51MUlHUXpSa0NcbjhwYzhJQTdqeGZuOWhuK3lRNnJCVkZ1Nm5Ieko3MUQxV2hYd3BzVCswb0pFTm9LRlRYOGZqWDBMazF0RVkzZ1hcbm56aXBDRUFkZCszUWIyL1BCOVdqcUJQZDRCUzdObFptQWxTSEloUEIrd0tCZ1FEenpQbGdYUGFtY0hOVDBDa0dcbm5MUXFtSmg5K3YvdWMxWHBnSmxVcjRsM2xLaVhGRGR1UVQ3cWhnQ0F5TXc4RmVXMS9IczUwVWpJWVVteUlFRC9cblg4Vld0NUg1NzMxbWgweGpSUG9uUlZPaTBPa0ExeWlBK08yOXNPYW9MTXBLK0sxUzJCSnhHcnVzKzh3NXV4K0hcblhlTjFQVFgrRmFTb1VieDN3YXZ3Ym1meWh3S0JnUURPbTZxVDF6OEdEczVIcDNKcjFTQVF1MTVGdXYzaWxnaHNcbkRpMU1WSndwcVhyenVsTjVSdy9pbnB4K3luVmprdGdkN1F5YmxTYUVjLzgwckM4V0pxdTd6WEpFV3M0Q0t0a0RcbmxUbGVuU2dMUnUrS08rcUNTbTNvRmtmanYxSnZLSFlpVHZCWWN0Yy8rbDdNMy83MzcvUVJ2QnEyQkJCeWVwVUZcbm4weG5rZ3JTNHdLQmdCQSsxVk10ZnFZQ2tqekFmeXRZbjh6QzFFNTR2anNXWm1BajJKUERDcWIwT2ZPdlBpNVRcbm5LeWw5enlkcExaUVF6bElOTEhhbHozNjlaMHY5d1ROVGVvRVcyN0xIWkVLYlBXa2NBTFQ1TW5SbitVVmNUWitcbjMrQ20vQWV5ekZ3SWpBd2NKOFp2b0pmYlEzV0pXWVY4cFI1MzBMUTRudnMwVHhtdnh4UkRWeHZiQW9HQUVXeVJcbll5eDQ5VExVZExpTGJzcW5qS0d3bnFMWmZIMTRzbHd6dDhjKzhFaUp4UHBHeGVpWFQxNWZCbFpldGdvUlRkektcbi9tZ0N1cExweC9CcGZDM0F0L0xvbXhrcFhJZHVpOTNPMjhyWE1MUkh5Vm1xT2xpNmtpTW01dThnclowMDhVbkVcbi9VQ2FKSndoMkpkZmNsdDdNdSt4TlA1OHdKclQ3SWgxc2hwZjdVOENnWUI1MUp2N1ZWeGdRMnEyeGNqWmV0VXVcbktLZ05Rb080WjJXK1hiRGgyTjkvMEhCNW1wSy9VMzhXSy9wclcyR2d0ZVIwS1d1MkcwTzdHYnVWbmF2Ly9yUzVcbjMvOEZDanFkMzI4bmp5YzdsZVhPTXROaERENGhxN05tcEgvY2tyZ0hrVXM2QXhMQS9vZ2dBbERta1JZQm1WWnZcblJrU1BDSlZJMzg5d3YvZjZ5Y2xqMVE9PVxuLS0tLS1FTkQgUFJJVkFURSBLRVktLS0tLSIsICJwcm9maWxlX2FybiI6ICJhcm46YXdzOnJvbGVzYW55d2hlcmU6ZXUtY2VudHJhbC0xOjMyNzI1NzUwMjE1MTpwcm9maWxlLzU0Yzc2ZmIzLWFlYzQtNDU5Ny1hOGE1LTU4M2U4OWUzMWY4NiIsICJyb2xlX2FybiI6ICJhcm46YXdzOmlhbTo6MzI3MjU3NTAyMTUxOnJvbGUvcGx1Z2luLXJvbGVzLWFueXdoZXJlLXJvbGUiLCAidHJ1c3RfYW5jaG9yX2FybiI6ICJhcm46YXdzOnJvbGVzYW55d2hlcmU6ZXUtY2VudHJhbC0xOjMyNzI1NzUwMjE1MTp0cnVzdC1hbmNob3IvY2YwYmE5NDctNjM2MC00MGRiLTk1YTUtOGEzMTQ4MWNlOGMzIiwgInJlZ2lvbiI6ICJldS1jZW50cmFsLTEifQ==") \
        .server_region('eu-central-1')\
        .description('GHT-testing plugin')\
        .icon('report')\
        .build()


'''    
        #for hosting plugin on the manufacturing server
    PluginClient.of() \
        .vendor('Zeto') \
        .name('GHT testing plugin') \
        .version('1.0.0') \
        .operation_pdf() \
        .plugin(MyFirstPlugin) \
        .auth_string("eyJjZXJ0aWZpY2F0ZSI6ICItLS0tLUJFR0lOIENFUlRJRklDQVRFLS0tLS1cbk1JSURjakNDQWxxZ0F3SUJBZ0lRTXA0eE81OFY4YU9QeE5EMS9jYU1rREFOQmdrcWhraUc5dzBCQVFzRkFEQVNcbk1SQXdEZ1lEVlFRRERBZGhjR2x3Y205a01CNFhEVEl6TVRFek1ERXlNamN5TmxvWERUSTJNRE13TkRFeU1qY3lcbk5sb3dRakZBTUQ0R0ExVUVBd3czY205aVpYSjBjMjkwYVVCNlpYUnZhVzVqTG1OdmJTeGFaWFJ2TEU5WWNIUkVcblYyVkllR001WlVGT2NFSlZTWEJtT1hJc1ZISjFaVENDQVNJd0RRWUpLb1pJaHZjTkFRRUJCUUFEZ2dFUEFEQ0NcbkFRb0NnZ0VCQU1DaXBLL0xvS1hqSS9ibnRhSlpWTW45RFIvdHFiOEhMYVV3RDJYL2dHbGpUeTNlQXlWWUlWWWVcbjBnYjY3Vmx1MWZHbWpYcVovU3dxOW95MWhTcU90Z2Q1VnFtVXl1Z08zYndVNHVMYnRkOGJJaVdYdnVIeER2em1cbjllV1p1cy9tWjVxT3B6cnRJeFNqRlZZTit3eHJLbzJyK2xENko0U2ZERUg3Uit3UXJBVmhIMzZGN0pBUDlKV3JcbkxqbGNWSi8wZS9HTExWMC9iS01yZXlzRThMSGNKSEE3cUtHbkVOQnRXbUpFSFM2MEV6TjRaQ1pYVXNIbG5kdG9cblVVbmxBNG40V2Z5WlJ6bGJrVGw4VGNPRG91MzVseElhazQ5RkhybVQrMGFWb3Fhd0svNkIyQXNvMEZ1U2pGemZcbkw2bzRXNTc2NlI1dHdjQVBaTTBnZUptVFFMZjZ0NThDQXdFQUFhT0JrekNCa0RBSkJnTlZIUk1FQWpBQU1CMEdcbkExVWREZ1FXQkJUdVJHVTFtTHhiUUZQemFyN3F4N1dldzdZM2lqQkNCZ05WSFNNRU96QTVnQlNLWUlQL0FVdCtcbk10SFhwU0lFTERqYzF4OXVDNkVXcEJRd0VqRVFNQTRHQTFVRUF3d0hZWEJwY0hKdlpJSUpBSmtWTkV2bWtSL0Fcbk1CTUdBMVVkSlFRTU1Bb0dDQ3NHQVFVRkJ3TUNNQXNHQTFVZER3UUVBd0lIZ0RBTkJna3Foa2lHOXcwQkFRc0ZcbkFBT0NBUUVBd2RhbzdkQTVXMFJBaWdOYXllWnN0Ump6VXp6TVpmdDUzQ21GOEFraFNKNU8vcm5pa0JWb0JxYUpcbkRjN0czaVh0RlNxMThTZkRxWm90VVJKWTc4ZmhlVE9JUEcrd2dOanBsdmZVcmNabVU0bkdreHVxdGhHazJ3QUdcbmFnRGtqNUdtMkhTclhtT1JRU2krNFpEcFM2MEgvckhZTmQ4UlB2RzlkcGVGMXVBdTNoa1htMS9YN1UyWTg1U3NcbkZHK0RrMUhaVXFwdkwyZUg3ejg2Y3BqM0dSMU84dHlxT0lCMUkvdytnbGxzVU9nNTJON2hQRjNpb0liYkMrbVRcbjZ5SnpJaDdlc2VoamdBZVEzb3JRV0dRdHNqeDhJN0Z4cDh0Mk9ZTkI5WTBSeHNGWEtTWHVQTDF1SDcwQlZFWTJcbnhmODI3QVZqS3NsMGQ3cWJ1SHZzdG1CTmZqT3kvQT09XG4tLS0tLUVORCBDRVJUSUZJQ0FURS0tLS0tIiwgInByaXZrZXkiOiAiLS0tLS1CRUdJTiBQUklWQVRFIEtFWS0tLS0tXG5NSUlFdlFJQkFEQU5CZ2txaGtpRzl3MEJBUUVGQUFTQ0JLY3dnZ1NqQWdFQUFvSUJBUURBb3FTdnk2Q2w0eVAyXG41N1dpV1ZUSi9RMGY3YW0vQnkybE1BOWwvNEJwWTA4dDNnTWxXQ0ZXSHRJRyt1MVpidFh4cG8xNm1mMHNLdmFNXG50WVVxanJZSGVWYXBsTXJvRHQyOEZPTGkyN1hmR3lJbGw3N2g4UTc4NXZYbG1iclA1bWVhanFjNjdTTVVveFZXXG5EZnNNYXlxTnEvcFEraWVFbnd4QiswZnNFS3dGWVI5K2hleVFEL1NWcXk0NVhGU2Y5SHZ4aXkxZFAyeWpLM3NyXG5CUEN4M0NSd082aWhweERRYlZwaVJCMHV0Qk16ZUdRbVYxTEI1WjNiYUZGSjVRT0orRm44bVVjNVc1RTVmRTNEXG5nNkx0K1pjU0dwT1BSUjY1ay90R2xhS21zQ3YrZ2RnTEtOQmJrb3hjM3krcU9GdWUrdWtlYmNIQUQyVE5JSGlaXG5rMEMzK3JlZkFnTUJBQUVDZ2dFQUM2elAxbWU5QU5EWGFTdDU4ckQvVm82anNQK3lmZ1Z0V3NtaTVVZXEvTGdzXG5JRmVJVVVjYUhpSVlFSnc4cm1MdFRJd1BueEtlS1lNaEZqNGFLQTRtTWlYRWZ0cXB3WTdGR1pVYWV5MWR4SHZjXG5nZkxFZXVVbFIyYW9HSlpNb0RVU3FtZGk4MHRVQ1BncFh1SFNDVnFsWlppNnZ4V3FOM3Q4UHJPQUZHcWVRbVVaXG5qSmJXTnJkNVg5TmNsL3R1QWFxbkZiV2ptZ0kyaGp6NVd1SE13VUVyMnU1cFdYRndBZjRSTXdpRzM5SllXYTRuXG5DdU9sME5IYWN1S01QUk55NE9SSFlEdWYvajdQMDhWVWRveUZLeU51eEUvRE1iVDBKbVFCOUZXSnNESXh0UTNwXG51Z3dKUU5PaHJ0WDVLOEJsejdSME5UR0JDbm5VSzc2dnp1eUQ0VytGQVFLQmdRRHRxdmNVTDAvaEZJTE5rblVGXG5yaXF4NUFSbWRZVEhnc3lkMTh0Mnd1U3ZCbVprT1UvU1NFSUNIYmc4bDF3cmxZdDlpbUtoRjJZZWZtR081cUEzXG4wYVpENktVem1yZ0FEUDU3ZGdzSDI2Z1lQM2NWOWFNUEt6dFBLRUtROHlDa05LbzRONVVscGhsSmNuYWRndmJkXG5tV3BkY3BncFdvYnQzMnBVRDRKd2VLanJZUUtCZ1FEUGZuVWtWSVpTZERoVm05RUNUa3d3b3IrcTA4TGNSSDdiXG55WngvSm9jOStwSzhTODE3cXhqVnYyaHZaSG91Z3ZQUW9QQkVFYVFuSllpLzFYOFFCQ3RsOXVMS0JscDVqQS83XG5UU1ppWUdUM08rcnZNb0J0TG03U1FHTXVOTHAyREdGZTNMNkI2NDQ5VjcxallTa29iSXBkS1FSZktrK1RzNFZ5XG5TdUg0VnRHQy93S0JnUUMxblE5eHZUV3RLWjlLdDcvMHQwS283VFR2bHA0QVYxTkV5c0lQM3A0aG9TSmRNKzVyXG5JZ3hPMGFjWHBoSW91LzM3ME9QTmRiUHpXVi96Y3dpN250a095NWh4OXFqa1lRbVdEbjRmWXhyd2JJN3ptT0VoXG5sa2VjRllmSWZBRlZlV2taekYrTWhZQ05QNHFra285U2h3bGduMURuU09ZU244Y0F0VmxYMk53OEFRS0JnQkZpXG5oQWdXT09iaHEvS29TbnZKK2FJOWtKZU1oSkFXQVJjRExtU000dG56aTZYUktCZExmNW94SGx2dTdEbkhhUXc2XG4zOGFrUDcrejZtQkFVQlFVZFZwbXRCdS9Lb2R5ajhnN2I1TGdoclVjUlJQamhGVWhoZEdCNlkxdWg0enFmcUlIXG5Gc01sN0ZZQmF2SFBxellpMmZqeVBkYUhZZ0Y3RWh4QVgreUJ3YTRiQW9HQUpwQ0YyOHhWK3BKaGFYQ3R5RnV6XG44V0RxVTFCcXV1ODFaeko0T1BYMjhOTmJyZ3l5eXFFR0gybFc0bzExeUxuSURNSTYyWGdSZlNQWCsyazFKcC9UXG56akY5R0N6SWFSemJxb1ZsakFnYTZlWUw0QUc4cUlIYmRLQ0NrZkZpeDJrYmJVTXdRV2pMWHdhUnQ0dEowMFl6XG52K3pWUEh4WjlWZ0xiQmMwV2I2SXp4OD1cbi0tLS0tRU5EIFBSSVZBVEUgS0VZLS0tLS0iLCAicHJvZmlsZV9hcm4iOiAiYXJuOmF3czpyb2xlc2FueXdoZXJlOnVzLWVhc3QtMTo0MTAzOTQzMDUyODc6cHJvZmlsZS8yZjNkMWNhNy0xNjUwLTRjODUtOWFiOC1lYTZhN2YxM2IxMmMiLCAicm9sZV9hcm4iOiAiYXJuOmF3czppYW06OjQxMDM5NDMwNTI4Nzpyb2xlL3BsdWdpbi1yb2xlcy1hbnl3aGVyZS1yb2xlIiwgInRydXN0X2FuY2hvcl9hcm4iOiAiYXJuOmF3czpyb2xlc2FueXdoZXJlOnVzLWVhc3QtMTo0MTAzOTQzMDUyODc6dHJ1c3QtYW5jaG9yLzk3OGE4MzkwLWFmZjgtNDFkOC05NWFlLTkwMGI2OTBmNzU5YSIsICJyZWdpb24iOiAidXMtZWFzdC0xIn0=") \
        .description('Zeto GHT plugin') \
        .icon('report') \
        .build()

   
''' 




        




        
        
if __name__ == '__main__':
    main()
