import numpy as np
import pyedflib

def generate_sine_wave(frequency, duration, sampling_rate):
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    sine_wave = np.sin(2 * np.pi * frequency * t)
    return sine_wave

# Parameters
frequency = 4  # 4 Hz
duration = 10  # 10 seconds
sampling_rate = 500  # 256 Hz
n_channels = 19  # Number of channels

# Generate sine wave for each of the 19 channels
sine_waves = [generate_sine_wave(frequency, duration, sampling_rate) for _ in range(n_channels)]

# EDF file parameters
file_name = 'sine_wave_4hz.edf'

# Create EDF writer
edf_writer = pyedflib.EdfWriter(file_name, n_channels, file_type=pyedflib.FILETYPE_EDFPLUS)

# Define channel info
channel_info = []
for i in range(n_channels):
    ch_dict = {
        'label': f'CH_{i+1}',
        'dimension': 'uV',
        'sample_rate': sampling_rate,
        'physical_min': -1.0,
        'physical_max': 1.0,
        'digital_min': -32768,
        'digital_max': 32767,
        'transducer': '',
        'prefilter': ''
    }
    channel_info.append(ch_dict)

edf_writer.setSignalHeaders(channel_info)

# Write data to EDF file
edf_writer.writeSamples(sine_waves)

# Close the EDF writer
edf_writer.close()

print(f'EDF file "{file_name}" created successfully.')

