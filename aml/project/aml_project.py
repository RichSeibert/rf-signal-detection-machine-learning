import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

samples = np.fromfile('FMcapture1.dat', np.complex64) # Read in file.  We have to tell it what format it is
print(samples)

plt.plot(np.log(np.abs(samples[:1000000])))
plt.show()
