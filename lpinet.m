lgraph = layerGraph();
tempLayers = [
    imageInputLayer([50 50 1],"Name","input")
    convolution2dLayer([7 7],64,"Name","conv_1","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1")
    eluLayer(1,"Name","elu_1")
    maxPooling2dLayer([3 3],"Name","maxpool_1","Padding","same","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 3],64,"Name","conv_2","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2")
    eluLayer(1,"Name","elu_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 1],64,"Name","conv_3","Padding","same")
    batchNormalizationLayer("Name","batchnorm_3")
    eluLayer(1,"Name","elu_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","depthcat_1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","conv_4","Padding","same")
    batchNormalizationLayer("Name","batchnorm_4")
    eluLayer(1,"Name","elu_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 3],64,"Name","conv_5","Padding","same")
    batchNormalizationLayer("Name","batchnorm_5")
    eluLayer(1,"Name","elu_5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 1],64,"Name","conv_6","Padding","same")
    batchNormalizationLayer("Name","batchnorm_6")
    eluLayer(1,"Name","elu_6")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(3,"Name","depthcat_2")
    convolution2dLayer([1 1],64,"Name","conv_7","Padding","same")
    batchNormalizationLayer("Name","batchnorm_7")
    eluLayer(1,"Name","elu_7")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_1")
    maxPooling2dLayer([3 3],"Name","maxpool_2","Padding","same","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 3],64,"Name","conv_8","Padding","same")
    batchNormalizationLayer("Name","batchnorm_8")
    eluLayer(1,"Name","elu_8")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 1],64,"Name","conv_9","Padding","same")
    batchNormalizationLayer("Name","batchnorm_9")
    eluLayer(1,"Name","elu_9")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","depthcat_3");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","conv_10","Padding","same")
    batchNormalizationLayer("Name","batchnorm_10")
    eluLayer(1,"Name","elu_10")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 3],64,"Name","conv_11","Padding","same")
    batchNormalizationLayer("Name","batchnorm_11")
    eluLayer(1,"Name","elu_11")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 1],64,"Name","conv_12","Padding","same")
    batchNormalizationLayer("Name","batchnorm_12")
    eluLayer(1,"Name","elu_12")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(3,"Name","depthcat_4")
    convolution2dLayer([1 1],64,"Name","conv_13","Padding","same")
    batchNormalizationLayer("Name","batchnorm_13")
    eluLayer(1,"Name","elu_13")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_2")
    maxPooling2dLayer([3 3],"Name","maxpool_3","Padding","same","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 3],64,"Name","conv_14","Padding","same")
    batchNormalizationLayer("Name","batchnorm_14")
    eluLayer(1,"Name","elu_14")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 1],64,"Name","conv_15","Padding","same")
    batchNormalizationLayer("Name","batchnorm_15")
    eluLayer(1,"Name","elu_15")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","depthcat_5");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","conv_16","Padding","same")
    batchNormalizationLayer("Name","batchnorm_16")
    eluLayer(1,"Name","elu_16")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 3],64,"Name","conv_17","Padding","same")
    batchNormalizationLayer("Name","batchnorm_17")
    eluLayer(1,"Name","elu_17")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 1],64,"Name","conv_18","Padding","same")
    batchNormalizationLayer("Name","batchnorm_18")
    eluLayer(1,"Name","elu_18")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(3,"Name","depthcat_6")
    convolution2dLayer([1 1],64,"Name","conv_19","Padding","same")
    batchNormalizationLayer("Name","batchnorm_19")
    eluLayer(1,"Name","elu_19")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","addition_3")
    globalAveragePooling2dLayer("Name","gapool")
    fullyConnectedLayer(64,"Name","fc_1")
    dropoutLayer(0.5,"Name","dropout")
    fullyConnectedLayer(13,"Name","fc_2")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];
lgraph = addLayers(lgraph,tempLayers);

lgraph = connectLayers(lgraph,"maxpool_1","conv_2");
lgraph = connectLayers(lgraph,"maxpool_1","conv_3");
lgraph = connectLayers(lgraph,"maxpool_1","addition_1/in2");
lgraph = connectLayers(lgraph,"elu_2","depthcat_1/in1");
lgraph = connectLayers(lgraph,"elu_3","depthcat_1/in2");
lgraph = connectLayers(lgraph,"depthcat_1","conv_4");
lgraph = connectLayers(lgraph,"depthcat_1","depthcat_2/in3");
lgraph = connectLayers(lgraph,"elu_4","conv_5");
lgraph = connectLayers(lgraph,"elu_4","conv_6");
lgraph = connectLayers(lgraph,"elu_5","depthcat_2/in1");
lgraph = connectLayers(lgraph,"elu_6","depthcat_2/in2");
lgraph = connectLayers(lgraph,"elu_7","addition_1/in1");
lgraph = connectLayers(lgraph,"maxpool_2","conv_8");
lgraph = connectLayers(lgraph,"maxpool_2","conv_9");
lgraph = connectLayers(lgraph,"maxpool_2","addition_2/in2");
lgraph = connectLayers(lgraph,"elu_8","depthcat_3/in1");
lgraph = connectLayers(lgraph,"elu_9","depthcat_3/in2");
lgraph = connectLayers(lgraph,"depthcat_3","conv_10");
lgraph = connectLayers(lgraph,"depthcat_3","depthcat_4/in3");
lgraph = connectLayers(lgraph,"elu_10","conv_11");
lgraph = connectLayers(lgraph,"elu_10","conv_12");
lgraph = connectLayers(lgraph,"elu_11","depthcat_4/in1");
lgraph = connectLayers(lgraph,"elu_12","depthcat_4/in2");
lgraph = connectLayers(lgraph,"elu_13","addition_2/in1");
lgraph = connectLayers(lgraph,"maxpool_3","conv_14");
lgraph = connectLayers(lgraph,"maxpool_3","conv_15");
lgraph = connectLayers(lgraph,"maxpool_3","addition_3/in2");
lgraph = connectLayers(lgraph,"elu_15","depthcat_5/in2");
lgraph = connectLayers(lgraph,"elu_14","depthcat_5/in1");
lgraph = connectLayers(lgraph,"depthcat_5","conv_16");
lgraph = connectLayers(lgraph,"depthcat_5","depthcat_6/in3");
lgraph = connectLayers(lgraph,"elu_16","conv_17");
lgraph = connectLayers(lgraph,"elu_16","conv_18");
lgraph = connectLayers(lgraph,"elu_18","depthcat_6/in2");
lgraph = connectLayers(lgraph,"elu_17","depthcat_6/in1");
lgraph = connectLayers(lgraph,"elu_19","addition_3/in1");