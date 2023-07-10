import os
import sys

path = ''
sys.path.append(os.path.abspath(path + 'WCRG'))
sys.path.append(os.path.abspath(path + 'WCRG/WCRG'))
sys.path.append(os.path.abspath(path + 'WCRG/Wavelet_Packets'))
sys.path.append(os.path.abspath(path + 'WCRG/WCRG/Models'))

from Wavelet_Packets import *
from WCRG import *
