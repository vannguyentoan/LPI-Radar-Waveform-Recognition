% This present the MATLAB codes for LPI radar signal generation and Choi-Williams distribution-based time-frequency image generation.

% The codes of Choi-Williams distribution have been achieved from the
% Time-Frequency Toolbox in http://tftb.nongnu.org/ 
% The TFTB is distributed under the terms of the GNU Public Licence.

addpath(genpath("\tftb-0.2")); % Modify the pathname in your pc

%% initial parameters configurations
fs = 100e6; % sample frequency
A = 1;      % amplitude
waveforms = {'Rect','LFM','Costas','Barker','Frank','P1','P2','P3','P4','T1','T2','T3','T4'};   % 13 LPI waveform codes
datasetCWD = 'dataset-CWD-50';

for i = 1 : length(waveforms)
    % create the folders for dataset storage
    mkdir(fullfile(datasetCWD,waveforms{i}));
end

fps = 1000;  % the number of signal per SNR per waveform codes
% filter configuration
g=kaiser(63,0.5);
h=kaiser(63,0.5);
imgSize = 50;

%% initial channels
atenuationset = -20:2:0; % path gain range
delayset = (1:50:1000)*1e-9; % path delay range
atenuations = [0,atenuationset(randperm(length(atenuationset),5))]; % path gains
delays = [0,delayset(randperm(length(delayset),5))]; % path delays
multipathChannel = comm.RayleighChannel(...
    'SampleRate', fs, ...
    'PathDelays', delays, ...
    'AveragePathGains', atenuations, ...
    'MaximumDopplerShift', randi([5 400]));

SNR = -20 : 1 : 10;     % snr range
    
for n = 22 : length(SNR)
    disp(['SNR = ',sprintf('%+02d',SNR(n))])
 
    for k = 1 : length(waveforms)
        waveform = waveforms{k};
        switch waveform
            case 'Rect'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc = fc(randperm(fps));
                N = linspace(512,1920,fps);
                N = round(N(randperm(fps)));
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1 : fps
                    release(multipathChannel);
                    atenuations = [0,atenuationset(randperm(length(atenuationset),5))]; % Uniformly random path gains generation
                    delays = [0,delayset(randperm(length(delayset),5))]; % Uniformly random path delays generation
                    multipathChannel.PathDelays = delays;
                    multipathChannel.AveragePathGains = atenuations;
                    multipathChannel.MaximumDopplerShift = randi([10 1000]);
                    wav = type_Rect(N(idx),fs,A,fc(idx));
                    wav = multipathChannel(wav');       % passing over multipath channels
                    wav = awgn(wav,SNR(n),'measured');
                    t=1:length(wav);
                    [CWD_TFD,~,~] = FTCWD(wav,t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD,fullfile(waveformfolderCWD,sprintf('rect-snr%02d-no%05d.png',n,idx)));
                    
                end
            case 'LFM'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));
                B = linspace(fs/20, fs/16, fps);
                B = B(randperm(fps));
                N = linspace(512,1920,fps);
                N=round(N(randperm(fps)));
                sweepDirections = {'Up','Down'};
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps
                    release(multipathChannel);
                    atenuations = [0,atenuationset(randperm(length(atenuationset),5))]; % path gains
                    delays = [0,delayset(randperm(length(delayset),5))]; % path delays
                    multipathChannel.PathDelays = delays;
                    multipathChannel.AveragePathGains = atenuations;
                    multipathChannel.MaximumDopplerShift = randi([10 1000]);
                    wav = type_LFM(N(idx),fs,A,fc(idx),B(idx),sweepDirections{randi(2)});
                    wav = multipathChannel(wav'); % adding multipath channels
                    wav = awgn(wav,SNR(n),'measured');
                    t=1:length(wav);
                    [CWD_TFD,~,~] = FTCWD(wav,t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD,fullfile(waveformfolderCWD,sprintf('lfm-snr%02d-no%05d.png',n,idx)));
                    
                end
            case 'Costas'
                disp(['Generating ',waveform, ' waveform ...']);
                Lc = [3, 4, 5, 6];
                fcmin = linspace(fs/30,fs/24,fps);
                fcmin=fcmin(randperm(fps));
                N = linspace(512,1920,fps);
                N=round(N(randperm(fps)));
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps
                    release(multipathChannel);
                    atenuations = [0,atenuationset(randperm(length(atenuationset),5))]; % path gains
                    delays = [0,delayset(randperm(length(delayset),5))]; % path delays
                    multipathChannel.PathDelays = delays;
                    multipathChannel.AveragePathGains = atenuations;
                    multipathChannel.MaximumDopplerShift = randi([10 1000]);
                    NumHop = randperm(Lc(randi(4)));
                    wav = type_Costas(N(idx), fs, A, fcmin(idx), NumHop);
                    wav = multipathChannel(wav'); % adding multipath channels
                    wav = awgn(wav,SNR(n),'measured');
                    t=1:length(wav);
                    [CWD_TFD,~,~] = FTCWD(wav,t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD,fullfile(waveformfolderCWD,sprintf('costas-snr%02d-no%05d.png',n,idx)));
                    
                end
                
            case 'Barker'
                disp(['Generating ',waveform, ' waveform ...']);
                Lc = [7,11,13];
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));
                Ncc = 20:24;
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps
                    release(multipathChannel);
                    atenuations = [0,atenuationset(randperm(length(atenuationset),5))]; % path gains
                    delays = [0,delayset(randperm(length(delayset),5))]; % path delays
                    multipathChannel.PathDelays = delays;
                    multipathChannel.AveragePathGains = atenuations;
                    multipathChannel.MaximumDopplerShift = randi([10 1000]);
                    Bar = Lc(randi(3));
                    if Bar == 7
                        phaseCode = [0 0 0 1 1 0 1]*pi;
                    elseif Bar == 11
                        phaseCode = [0 0 0 1 1 1 0 1 1 0 1]*pi;
                    elseif Bar == 13
                        phaseCode = [0 0 0 0 0 1 1 0 0 1 0 1 0]*pi;
                    end
                    wav = type_Barker(Ncc, fs, A, fc(idx), phaseCode);
                    wav = multipathChannel(wav'); % adding multipath channels
                    wav = awgn(wav,SNR(n),'measured');
                    t=1:length(wav);
                    [CWD_TFD,~,~] = FTCWD(wav,t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD,fullfile(waveformfolderCWD,sprintf('barker-snr%02d-no%05d.png',n,idx)));
                    
                end
            case 'Frank'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));
                Ncc = [3,4,5];
                M = [6, 7, 8];
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps
                    release(multipathChannel);
                    atenuations = [0,atenuationset(randperm(length(atenuationset),5))]; % path gains
                    delays = [0,delayset(randperm(length(delayset),5))]; % path delays
                    multipathChannel.PathDelays = delays;
                    multipathChannel.AveragePathGains = atenuations;
                    multipathChannel.MaximumDopplerShift = randi([10 1000]);
                    wav = type_Frank(Ncc(randi(3)), fs, A, fc(idx), M(randi(3)));
                    wav = multipathChannel(wav'); % adding multipath channels
                    wav = awgn(wav,SNR(n),'measured');
                    t=1:length(wav);
                    [CWD_TFD,~,~] = FTCWD(wav,t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD,fullfile(waveformfolderCWD,sprintf('frank-snr%02d-no%05d.png',n,idx)));
                    
                end
            case 'P1'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));
                Ncc = [3,4,5];
                M = [6, 7, 8];
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps
                    release(multipathChannel);
                    atenuations = [0,atenuationset(randperm(length(atenuationset),5))]; % path gains
                    delays = [0,delayset(randperm(length(delayset),5))]; % path delays
                    multipathChannel.PathDelays = delays;
                    multipathChannel.AveragePathGains = atenuations;
                    multipathChannel.MaximumDopplerShift = randi([10 1000]);
                    wav = type_P1(Ncc(randi(3)), fs, A, fc(idx), M(randi(3)));
                    wav = multipathChannel(wav'); % adding multipath channels
                    wav = awgn(wav,SNR(n),'measured');
                    t=1:length(wav);
                    [CWD_TFD,~,~] = FTCWD(wav,t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD,fullfile(waveformfolderCWD,sprintf('p1-snr%02d-no%05d.png',n,idx)));

                end
            case 'P2'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));
                Ncc = [3,4,5];
                M = [6, 8];
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps
                    release(multipathChannel);
                    atenuations = [0,atenuationset(randperm(length(atenuationset),5))]; % path gains
                    delays = [0,delayset(randperm(length(delayset),5))]; % path delays
                    multipathChannel.PathDelays = delays;
                    multipathChannel.AveragePathGains = atenuations;
                    multipathChannel.MaximumDopplerShift = randi([10 1000]);
                    wav = type_P2(Ncc(randi(3)), fs, A, fc(idx), M(randi(2)));
                    wav = multipathChannel(wav'); % adding multipath channels
                    wav = awgn(wav,SNR(n),'measured');
                    t=1:length(wav);
                    [CWD_TFD,~,~] = FTCWD(wav,t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD,fullfile(waveformfolderCWD,sprintf('p2-snr%02d-no%05d.png',n,idx)));

                end
            case 'P3'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));
                Ncc = [3,4,5];
                p = [36, 49, 64];
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps
                    release(multipathChannel);
                    atenuations = [0,atenuationset(randperm(length(atenuationset),5))]; % path gains
                    delays = [0,delayset(randperm(length(delayset),5))]; % path delays
                    multipathChannel.PathDelays = delays;
                    multipathChannel.AveragePathGains = atenuations;
                    multipathChannel.MaximumDopplerShift = randi([10 1000]);
                    wav = type_P3(Ncc(randi(3)), fs, A, fc(idx), p(randi(3)));
                    wav = multipathChannel(wav'); % adding multipath channels
                    wav = awgn(wav,SNR(n),'measured');
                    t=1:length(wav);
                    [CWD_TFD,~,~] = FTCWD(wav,t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD,fullfile(waveformfolderCWD,sprintf('p3-snr%02d-no%05d.png',n,idx)));

                end
                
            case 'P4'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));
                Ncc = [3,4,5];
                p = [36, 49, 64];
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps
                    release(multipathChannel);
                    atenuations = [0,atenuationset(randperm(length(atenuationset),5))]; % path gains
                    delays = [0,delayset(randperm(length(delayset),5))]; % path delays
                    multipathChannel.PathDelays = delays;
                    multipathChannel.AveragePathGains = atenuations;
                    multipathChannel.MaximumDopplerShift = randi([10 1000]);
                    wav = type_P4(Ncc(randi(3)), fs, A, fc(idx), p(randi(3)));
                    wav = multipathChannel(wav'); % adding multipath channels
                    wav = awgn(wav,SNR(n),'measured');
                    t=1:length(wav);
                    [CWD_TFD,~,~] = FTCWD(wav,t,1024,g,h,1,0,imgSize);                   
                    imwrite(CWD_TFD,fullfile(waveformfolderCWD,sprintf('p4-snr%02d-no%05d.png',n,idx)));

                end
            case 'T1'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));
                Ng = [4,5,6];
                N = linspace(512,1920,fps);
                N=round(N(randperm(fps)));
                Nps = 2;
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps
                    release(multipathChannel);
                    atenuations = [0,atenuationset(randperm(length(atenuationset),5))]; % path gains
                    delays = [0,delayset(randperm(length(delayset),5))]; % path delays
                    multipathChannel.PathDelays = delays;
                    multipathChannel.AveragePathGains = atenuations;
                    multipathChannel.MaximumDopplerShift = randi([10 1000]);
                    wav = type_T1(fs, A, fc(idx),Nps,Ng(randi(3)));
                    wav = multipathChannel(wav'); % adding multipath channels
                    wav = awgn(wav,SNR(n),'measured');
                    t=1:length(wav);
                    [CWD_TFD,~,~] = FTCWD(wav,t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD,fullfile(waveformfolderCWD,sprintf('t1-snr%02d-no%05d.png',n,idx)));

                end
                
            case 'T2'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));
                Ng = [4,5,6];
                Nps = 2;
                N = linspace(512,1920,fps);
                N=round(N(randperm(fps)));
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps
                    release(multipathChannel);
                    atenuations = [0,atenuationset(randperm(length(atenuationset),5))]; % path gains
                    delays = [0,delayset(randperm(length(delayset),5))]; % path delays
                    multipathChannel.PathDelays = delays;
                    multipathChannel.AveragePathGains = atenuations;
                    multipathChannel.MaximumDopplerShift = randi([10 1000]);
                    wav = type_T2(fs, A, fc(idx),Nps,Ng(randi(3)));
                    wav = multipathChannel(wav'); % adding multipath channels
                    wav = awgn(wav,SNR(n),'measured');
                    t=1:length(wav);
                    [CWD_TFD,~,~] = FTCWD(wav,t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD,fullfile(waveformfolderCWD,sprintf('t2-snr%02d-no%05d.png',n,idx)));

                end
                
            case 'T3'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));
                B = linspace(fs/20,fs/10,fps);
                B = B(randperm(fps));
                Ng = [4,5,6];
                N = linspace(512,1920,fps);
                N=round(N(randperm(fps)));
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps
                    release(multipathChannel);
                    atenuations = [0,atenuationset(randperm(length(atenuationset),5))]; % path gains
                    delays = [0,delayset(randperm(length(delayset),5))]; % path delays
                    multipathChannel.PathDelays = delays;
                    multipathChannel.AveragePathGains = atenuations;
                    multipathChannel.MaximumDopplerShift = randi([10 1000]);
                    wav = type_T3(N(idx), fs, A, fc(idx), Nps,B(idx));
                    wav = multipathChannel(wav'); % adding multipath channels
                    wav = awgn(wav,SNR(n),'measured');
                    t=1:length(wav);
                    [CWD_TFD,~,~] = FTCWD(wav,t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD,fullfile(waveformfolderCWD,sprintf('t3-snr%02d-no%05d.png',n,idx)));

                end
                
            case 'T4'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/5,fps);
                fc=fc(randperm(fps));
                B = linspace(fs/20,fs/10,fps);
                B = B(randperm(fps));
                Ng = [4,5,6];
                N = linspace(512,1920,fps);
                N=round(N(randperm(fps)));
                waveformfolderCWD = fullfile(datasetCWD,waveform);
                for idx = 1:fps
                    release(multipathChannel);
                    atenuations = [0,atenuationset(randperm(length(atenuationset),5))]; % path gains
                    delays = [0,delayset(randperm(length(delayset),5))]; % path delays
                    multipathChannel.PathDelays = delays;
                    multipathChannel.AveragePathGains = atenuations;
                    multipathChannel.MaximumDopplerShift = randi([10 1000]);
                    wav = type_T4(N(idx), fs, A, fc(idx), Nps,B(idx));
                    wav = multipathChannel(wav');       % adding multipath channels
                    wav = awgn(wav,SNR(n),'measured');
                    t=1:length(wav);
                    [CWD_TFD,~,~] = FTCWD(wav,t,1024,g,h,1,0,imgSize);
                    imwrite(CWD_TFD,fullfile(waveformfolderCWD,sprintf('t4-snr%02d-no%05d.png',n,idx)));
                    
                end
                
            otherwise
                disp('Done!')
        end
        
    end
    
end


