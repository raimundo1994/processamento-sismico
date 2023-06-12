import numpy as np
import matplotlib.pyplot as plt
import obspy

def raw_wiggle_plot(stream, dt,title = "original data",figsize=(10, 6),filename = None):
    """
    Generate a wiggle plot to display seismic traces as waves.

    Parameters:
        - traces (obspy.core.stream.Stream): Stream object containing seismic traces.
        - dt (float): Sampling interval in seconds.
        - figsize (tuple, optional): Figure size (width, height) in inches. Default is (10, 6).
        - filename (string): Picture name to save 

    Returns:
        None

    Note:
        This function requires the following libraries to be installed: obspy, numpy, matplotlib
    """
    
    num_traces = len(stream)
    num_samples = stream[0].stats.npts

    plt.figure(figsize=figsize)
    for i in range(num_traces):
        amplitudes = stream[i].data
        offset = i # Adjust the vertical offset between traces
            
        # Finding the final index where the value of normalized_amplitudes is different from zero.
        end_index = np.where(amplitudes != 0)[0]
        if len(end_index) == 0:
            continue
        end_index = end_index[-1]

        # Calculate the time values for the x-axis
        times = np.arange(num_samples)*dt

        # Plotting positive and negative amplitudes as filled polygons.
        plt.fill_betweenx(times, offset - amplitudes[:end_index + 1],
                          offset, where= amplitudes[:end_index + 1] >= 0,
                          facecolor='black', linewidth=0.5, alpha=0.5)
        plt.fill_betweenx(times, offset - amplitudes[:end_index + 1],
                          offset, where= amplitudes[:end_index + 1] < 0,
                          facecolor='black', linewidth=0.5, alpha=0.3)
      
    plt.gca().invert_yaxis()
    plt.xlabel('Trace Number',fontsize=12,weight='bold', alpha=.8)
    plt.ylabel('Time (s)',fontsize=12,weight='bold', alpha=.8)
    plt.title(title,fontsize=15,weight='bold', alpha=.8)
   
    plt.savefig(filename,dpi = 400, bbox_inches = 'tight')
    plt.show()
        
    
def normalized_wiggle_plot(stream, dt,title = " normalized data",figsize=(10, 6),filename = None):
    """
    Generate a normalized wiggle plot  to display seismic traces as waves.

    Parameters:
        - traces (obspy.core.stream.Stream): Stream object containing seismic traces.
        - dt (float): Sampling interval in seconds.
        - figsize (tuple, optional): Figure size (width, height) in inches. Default is (10, 6).
        - filename (string): Picture name to save 

    Returns:
        None

    Note:
        This function requires the following libraries to be installed: obspy, numpy, matplotlib
    """
    
    num_traces = len(stream)
    num_samples = stream[0].stats.npts

    plt.figure(figsize=figsize)
    for i in range(num_traces):
        amplitudes = stream[i].data
        offset = i # Adjust the vertical offset between traces
        
        # Calculating the maximum absolute amplitude.
        max_amplitude = max(abs(amplitudes))
        
        #Normalizing the amplitudes if the maximum amplitude is different from zero
        if max_amplitude != 0:
            normalized_amplitudes = amplitudes / max_amplitude
        else:
            normalized_amplitudes = np.zeros_like(amplitudes)
            
        # Finding the final index where the value of normalized_amplitudes is different from zero.
        end_index = np.where(normalized_amplitudes != 0)[0]
        if len(end_index) == 0:
            continue
        end_index = end_index[-1]

        # Calculate the time values for the x-axis
        times = np.arange(num_samples)*dt

        # Plotting positive and negative amplitudes as filled polygons.
        plt.fill_betweenx(times, offset - normalized_amplitudes[:end_index + 1],
                          offset, where=normalized_amplitudes[:end_index + 1] >= 0,
                          color='black', linewidth=0.5, alpha=0.5)
        plt.fill_betweenx(times, offset - normalized_amplitudes[:end_index + 1],
                          offset, where=normalized_amplitudes[:end_index + 1] < 0,
                          color='black', linewidth=0.5, alpha=0.3)

    plt.gca().invert_yaxis()
    plt.xlabel('Trace Number',fontsize=12,weight='bold', alpha=.8)
    plt.ylabel('Time (s)',fontsize=12,weight='bold', alpha=.8)
    plt.title(title,fontsize=15,weight='bold', alpha=.8)
    
    plt.savefig(filename,dpi = 400, bbox_inches = 'tight')
    plt.show()

def frequency_filter(data,
                     lowcut,
                     highcut,
                     start_time,
                     sampling_freq,
                     num_traces,
                     order=4):
    """
    Apply a frequency filter to seismic data.

    Parameters:
        data (ndarray): The seismic data as a numpy array.
        
        lowcut (float): The lower frequency cutoff in Hz.
        
        highcut (float): The upper frequency cutoff in Hz.
        
        start_time: Starting time of the data.
        
        sampling_freq (float): Frequency at which samples are taken or 
        recorded in a time series, such as seismic data in Hz
        
        num_traces(int): Total number of traces
        
        order = degree of the polynomial function used in the calculation of the filter.

    Returns:
        ndarray: The filtered seismic data as a numpy array.

    Notes:
        This function uses the ObsPy library to apply a bandpass filter to the seismic data.
        It transforms the data to the frequency domain using the Fourier transform, applies
        the filter, and then transforms the filtered data back to the time domain.

    """
    
    # Criar uma Stream de obspy a partir dos dados
    stream = obspy.Stream([obspy.Trace(data=data[i],
                                       header={'starttime': start_time,
                                               'sampling_rate': sampling_freq}) for i in range(num_traces)])

    # Aplicar a filtragem em cada traço
    filtered_data = np.zeros_like(data)
                          
    for i in range(num_traces):
        trace = stream[i]
        trace.detrend(type='linear')  # Remover tendência linear
        trace.filter('bandpass', freqmin=lowcut, freqmax=highcut, corners=order, zerophase=True)  # Filtragem passa-banda
        filtered_data[i] = trace.data
    
    return filtered_data


def plot_outliers_scatter(data,filename = None):
    num_cols = data.shape[1]
    x = np.arange(num_cols)
    for i in range(num_cols):
        plt.scatter(x[i] * np.ones_like(data[:, i]), data[:, i], c='b', alpha=0.5)
    plt.xlabel('Number of samples')
    plt.ylabel('Amplitudes')
    plt.title('Outliers Scatter Plot')
    plt.savefig(filename,dpi = 400, bbox_inches = 'tight')
    plt.show()