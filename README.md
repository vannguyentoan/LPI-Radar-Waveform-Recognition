# LPI-Radar-Waveform-Recognition
Automotive radars, with a widespread emergence in the last decade, have faced various jamming attacks. Utilizing low probability of intercept (LPI) radar waveforms, as one of the essential solutions, demands an accurate waveform recognizer at the intercept receiver. Numerous conventional approaches have been studied for LPI radar waveform recognition, but their performance is inadequate under channel condition deterioration. By exploiting deep learning (DL) to capture intrinsic radio characteristics, we develop a convolutional neural network (CNN), namely LPI-Net, for automatic radar waveform recognition. In particular, radar signals are first analyzed by a time-frequency analysis using the Choi-Williams distribution. Subsequently, LPI-Net, primarily consisting of three sophisticated modules, is built to learn the representational features of time-frequency images, in which each module is constructed with a preceding maps collection to gain feature diversity and a skip-connection to maintain informative identity.

This repository contains the end-to-end MATLAB codes of the LPI radar waveform recognition, which includes:
1. The MATLAB codes of Choi-Williams distribution which are referred from the Time-Frequency Toolbox at http://tftb.nongnu.org/ (The TFTB is distributed under the terms of the GNU Public Licence.) 
2. The MATLAB codes of radar waveform generation and time-frequency image transformation.
3. The MATLAB codes of deep deep network development.
4. The MATLAB codes of training model and performance evaluation.
