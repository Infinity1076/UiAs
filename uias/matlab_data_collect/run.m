% Configure one acquisition session and start both modality workers.

scriptDirectory = fileparts(mfilename('fullpath'));
outputRoot = fullfile(scriptDirectory, 'data');
sampleCount = 100;
startId = 1;

acquire_mmwave_acoustic(outputRoot, sampleCount, startId);
