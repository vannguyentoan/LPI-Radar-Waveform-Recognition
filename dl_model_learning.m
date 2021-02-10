imds = imageDatastore('dataset-CWD-50','IncludeSubfolders',true,'LabelSource','foldernames');
[imdsTrain,imdsTest] = splitEachLabel(imds,0.7,'randomized');
imdsTrain.Labels = categorical(imdsTrain.Labels);
imdsTest.Labels = categorical(imdsTest.Labels);



batchSize   = 256;
ValFre      = fix(length(imdsTrain.Files)/batchSize)
options = trainingOptions('sgdm', ...
    'MiniBatchSize',batchSize, ...
    'MaxEpochs',20, ...
    'Shuffle','every-epoch',...
    'InitialLearnRate',0.01, ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',8,...
    'LearnRateDropFactor',0.1,...
    'ValidationData',imdsTest, ...
    'ValidationFrequency',ValFre, ...
    'ValidationPatience',Inf, ...
    'Verbose',true ,...
    'VerboseFrequency',ValFre,...
    'Plots','training-progress',...
    'ExecutionEnvironment','multi-gpu');
trainednet = trainNetwork(imdsTrain,lgraph,options);

YPred = classify(trainednet,imdsTest,'MiniBatchSize',128,'ExecutionEnvironment','gpu');
YTest = imdsTest.Labels;
accuracy = sum(YPred == YTest)/numel(YTest)

plotconfusion(YTest,YPred)

result1FlowNetv1_1Size200_CWD = {};
result1FlowNetv1_1Size200_CWD{1,1} = trainednet;
result1FlowNetv1_1Size200_CWD{1,2} = YTest;
result1FlowNetv1_1Size200_CWD{1,3} = YPred;
result1FlowNetv1_1Size200_CWD{1,4} = imdsTest.Files;
result1FlowNetv1_1Size200_CWD{1,5} = accuracy;
filename = strcat('result1FlowNetv1_1Size200_CWD.mat');
save(filename,'result1FlowNetv1_1Size200_CWD')

conf_all_snr = zeros(13,13,31);
FileName = imdsTest.Files;
for i = 1 : length(YTest)
    codeSNR = str2double(FileName{i,1}(end-13:end-12));       
    yT = double(YTest(i,1));
    yP = double(YPred(i,1));
    
    if yT == yP
        conf_all_snr(yT,yT,codeSNR) = conf_all_snr(yT,yT,codeSNR) + 1;
    else
        conf_all_snr(yT,yP,codeSNR) = conf_all_snr(yT,yP,codeSNR) + 1;
    end
end

acc_13waveform_all_snr = zeros(13,31);
for k = 1 : 31
    for i = 1 : 13
        acc_13waveform_all_snr(i,k) = conf_all_snr(i,i,k)/sum(conf_all_snr(i,:,k));
    end
end
acc_13waveform_all_snr = acc_13waveform_all_snr * 100;
acc_avg_all_anr = zeros(1,31);
for k = 1 : 31
    allcorrection = 0;
    for i = 1 : 13
        allcorrection = allcorrection + conf_all_snr(i,i,k);
    end
    acc_avg_all_anr(1,k) = allcorrection/sum(sum(conf_all_snr(:,:,k)))*100;
end


conf = confusionmat(YTest,YPred);
classes = {'Barker','Costas','Frank','LFM','P1','P2','P3','P4','Rect','T1','T2','T3','T4'};

set(0,'DefaultAxesFontName', 'Times New Roman')
set(0,'DefaultAxesFontSize', 14)
set(0,'DefaultTextFontname', 'Times New Roman')
set(0,'DefaultTextFontSize', 14)
figure;
cm = confusionchart(conf,classes,'Normalization','row-normalized');
cm.FontName = 'Times New Roman';
cm.FontSize = 12;
cm.GridVisible = 'off';
cm.FontColor =[0 0 0];
colorbar = [0 0.3470 0.8410];
cm.DiagonalColor = colorbar;
cm.OffDiagonalColor = colorbar;

fig = gcf;
fig.PaperPositionMode = 'auto';
print('com-mag-confusion_mat','-depsc','-r600')
print('com-mag-confusion_mat','-dpng','-r600')


%%
load result1FlowNetv1_1Size200_CWD.mat

trainednet  = result1FlowNetv1_1Size200_CWD{1,1};
YTest       = result1FlowNetv1_1Size200_CWD{1,2};
YPred       = result1FlowNetv1_1Size200_CWD{1,3};
imdsTest.Files = result1FlowNetv1_1Size200_CWD{1,4};
accuracy    = result1FlowNetv1_1Size200_CWD{1,5};

plotconfusion(YTest,YPred)

conf_all_snr = zeros(13,13,31);
FileName = imdsTest.Files;
for i = 1 : length(YTest)
    codeSNR = str2double(FileName{i,1}(end-13:end-12));       
    yT = double(YTest(i,1));
    yP = double(YPred(i,1));
    
    if yT == yP
        conf_all_snr(yT,yT,codeSNR) = conf_all_snr(yT,yT,codeSNR) + 1;
    else
        conf_all_snr(yT,yP,codeSNR) = conf_all_snr(yT,yP,codeSNR) + 1;
    end
end

acc_13waveform_all_snr = zeros(13,31);
for k = 1 : 31
    for i = 1 : 13
        acc_13waveform_all_snr(i,k) = conf_all_snr(i,i,k)/sum(conf_all_snr(i,:,k));
    end
end
acc_13waveform_all_snr = acc_13waveform_all_snr * 100;
acc_avg_all_anr = zeros(1,31);
for k = 1 : 31
    allcorrection = 0;
    for i = 1 : 13
        allcorrection = allcorrection + conf_all_snr(i,i,k);
    end
    acc_avg_all_anr(1,k) = allcorrection/sum(sum(conf_all_snr(:,:,k)))*100;
end


conf = confusionmat(YTest,YPred);
classes = {'Barker','Costas','Frank','LFM','P1','P2','P3','P4','Rect','T1','T2','T3','T4'};

set(0,'DefaultAxesFontName', 'Times New Roman')
set(0,'DefaultAxesFontSize', 14)
set(0,'DefaultTextFontname', 'Times New Roman')
set(0,'DefaultTextFontSize', 14)
figure;
cm = confusionchart(conf,classes,'Normalization','row-normalized');
cm.FontName = 'Times New Roman';
cm.FontSize = 12;
cm.GridVisible = 'off';
cm.FontColor =[0 0 0];
colorbar = [0 0.3470 0.8410];
cm.DiagonalColor = colorbar;
cm.OffDiagonalColor = colorbar;

fig = gcf;
fig.PaperPositionMode = 'auto';
print('com-mag-confusion_mat','-depsc','-r600')
print('com-mag-confusion_mat','-dpng','-r600')

%%

load result1FlowNetv1_1Size200_WVD.mat

trainednet  = result1FlowNetv1_1Size200_WVD{1,1};
YTest       = result1FlowNetv1_1Size200_WVD{1,2};
YPred       = result1FlowNetv1_1Size200_WVD{1,3};
imdsTest.Files = result1FlowNetv1_1Size200_WVD{1,4};
accuracy    = result1FlowNetv1_1Size200_WVD{1,5};

plotconfusion(YTest,YPred)

conf_all_snr = zeros(13,13,31);
FileName = imdsTest.Files;
for i = 1 : length(YTest)
    codeSNR = str2double(FileName{i,1}(end-13:end-12));       
    yT = double(YTest(i,1));
    yP = double(YPred(i,1));
    
    if yT == yP
        conf_all_snr(yT,yT,codeSNR) = conf_all_snr(yT,yT,codeSNR) + 1;
    else
        conf_all_snr(yT,yP,codeSNR) = conf_all_snr(yT,yP,codeSNR) + 1;
    end
end

acc_13waveform_all_snr = zeros(13,31);
for k = 1 : 31
    for i = 1 : 13
        acc_13waveform_all_snr(i,k) = conf_all_snr(i,i,k)/sum(conf_all_snr(i,:,k));
    end
end
acc_13waveform_all_snr = acc_13waveform_all_snr * 100;
acc_avg_all_anr = zeros(1,31);
for k = 1 : 31
    allcorrection = 0;
    for i = 1 : 13
        allcorrection = allcorrection + conf_all_snr(i,i,k);
    end
    acc_avg_all_anr(1,k) = allcorrection/sum(sum(conf_all_snr(:,:,k)))*100;
end


conf = confusionmat(YTest,YPred);
classes = {'Barker','Costas','Frank','LFM','P1','P2','P3','P4','Rect','T1','T2','T3','T4'};

set(0,'DefaultAxesFontName', 'Times New Roman')
set(0,'DefaultAxesFontSize', 14)
set(0,'DefaultTextFontname', 'Times New Roman')
set(0,'DefaultTextFontSize', 14)
figure;
cm = confusionchart(conf,classes,'Normalization','row-normalized');
cm.FontName = 'Times New Roman';
cm.FontSize = 12;
cm.GridVisible = 'off';
cm.FontColor =[0 0 0];
colorbar = [0 0.3470 0.8410];
cm.DiagonalColor = colorbar;
cm.OffDiagonalColor = colorbar;

fig = gcf;
fig.PaperPositionMode = 'auto';
print('com-mag-confusion_mat','-depsc','-r600')
print('com-mag-confusion_mat','-dpng','-r600')