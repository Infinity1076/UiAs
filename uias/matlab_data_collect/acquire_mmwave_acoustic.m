function acquire_mmwave_acoustic(outputRoot, sampleCount, startId)
%ACQUIRE_MMWAVE_ACOUSTIC Collect gated acoustic and mmWave samples.
%
%   acquire_mmwave_acoustic("data", 10, 1)
%
% The coordinator implements a software gate for both modalities:
%
%   ARM -> AC/MM ARMED -> one shared GO -> AC/MM DONE -> next sample
%
% Two local MATLAB process workers own the two hardware devices. They begin
% acquisition only after the coordinator publishes the same GO marker. The
% coordinator does not issue the next ARM until both modalities have saved
% their MAT files and published DONE. The resulting files are:
%
%   <outputRoot>/mm/2d/<sampleId>.mat  variable: b             [128, 128]
%   <outputRoot>/mm/ra/<sampleId>.mat  variable: doa_2d_db     [50, 128]
%   <outputRoot>/ac/<sampleId>.mat     variable: energiess     [401, 1]
%
% Requirements:
%   - MATLAB R2024b
%   - Parallel Computing Toolbox (two process workers)
%   - Audio Toolbox
%   - Instrument Control Toolbox
%   - TI mmWave support package for IWR1843BOOST/DCA1000

    if nargin < 1 || strlength(string(outputRoot)) == 0
        outputRoot = fullfile(pwd, 'data');
    end
    if nargin < 2 || isempty(sampleCount)
        sampleCount = 1;
    end
    if nargin < 3 || isempty(startId)
        startId = 1;
    end

    outputRoot = char(java.io.File(char(outputRoot)).getCanonicalPath());
    validateattributes(sampleCount, {'numeric'}, ...
        {'scalar', 'integer', 'positive', 'finite'});
    validateattributes(startId, {'numeric'}, ...
        {'scalar', 'integer', 'positive', 'finite'});

    cfg = make_config(outputRoot);
    create_directories(cfg);
    clean_control_directory(cfg.controlDir);

    if ~license('test', 'Distrib_Computing_Toolbox')
        error(['Parallel Computing Toolbox is required because the acoustic ' ...
            'and mmWave devices must run concurrently.']);
    end

    pool = gcp('nocreate');
    if isempty(pool)
        pool = parpool('Processes', 2);
    elseif pool.NumWorkers < 2
        error('The active parallel pool must contain at least two workers.');
    end

    sessionToken = ['matlab-' char(java.util.UUID.randomUUID())];
    atomic_write_text(cfg.tokenPath, sessionToken);
    stopPath = fullfile(cfg.controlDir, ['stop_' sessionToken '.txt']);

    acousticFuture = parfeval(pool, @acoustic_worker, 0, cfg, sessionToken);
    mmwaveFuture = parfeval(pool, @mmwave_worker, 0, cfg, sessionToken);
    futures = [acousticFuture, mmwaveFuture];
    shutdown = onCleanup(@() stop_workers(stopPath, futures));

    readyPaths = {
        fullfile(cfg.controlDir, 'ac_worker.ready'), ...
        fullfile(cfg.controlDir, 'mm_worker.ready') ...
    };
    fprintf('Waiting for acoustic and mmWave workers...\n');
    wait_for_markers(readyPaths, sessionToken, cfg.workerTimeoutSeconds, futures);
    fprintf('Both workers are ready. Session: %s\n', sessionToken);

    session = struct();
    session.protocol = 'ARM-BOTH_ARMED-SHARED_GO-BOTH_DONE';
    session.sessionToken = sessionToken;
    session.outputRoot = outputRoot;
    session.startId = startId;
    session.sampleCount = sampleCount;
    session.startedUtc = utc_now();
    session.samples = struct([]);
    manifestPath = fullfile(outputRoot, 'acquisition_manifest.json');

    for offset = 0:sampleCount-1
        sampleId = startId + offset;
        expectedValue = sprintf('%s:%d', sessionToken, sampleId);
        armPath = fullfile(cfg.controlDir, ...
            sprintf('arm_%d_%s.json', sampleId, sessionToken));
        goPath = fullfile(cfg.controlDir, ...
            sprintf('go_%d_%s.txt', sampleId, sessionToken));
        armedPaths = {
            fullfile(cfg.controlDir, sprintf('ac_%d.armed', sampleId)), ...
            fullfile(cfg.controlDir, sprintf('mm_%d.armed', sampleId)) ...
        };
        donePaths = {
            fullfile(cfg.controlDir, sprintf('ac_%d.done', sampleId)), ...
            fullfile(cfg.controlDir, sprintf('mm_%d.done', sampleId)) ...
        };

        request = struct( ...
            'sampleId', sampleId, ...
            'sessionToken', sessionToken, ...
            'phase', 'ARM');
        atomic_write_text(armPath, jsonencode(request));
        armUtc = utc_now();

        wait_for_markers(armedPaths, expectedValue, ...
            cfg.roundTimeoutSeconds, futures);
        atomic_write_text(goPath, expectedValue);
        goUtc = utc_now();
        fprintf('[%d/%d] GO sample %d\n', offset + 1, sampleCount, sampleId);

        wait_for_markers(donePaths, expectedValue, ...
            cfg.roundTimeoutSeconds, futures);
        verify_sample_files(cfg, sampleId);
        doneUtc = utc_now();

        sampleRecord = struct( ...
            'sampleId', sampleId, ...
            'armUtc', armUtc, ...
            'goUtc', goUtc, ...
            'barrierDoneUtc', doneUtc);
        if isempty(session.samples)
            session.samples = sampleRecord;
        else
            session.samples(end + 1) = sampleRecord;
        end
        session.updatedUtc = doneUtc;
        atomic_write_text(manifestPath, jsonencode(session, PrettyPrint=true));

        delete_if_exists(armPath);
        delete_if_exists(goPath);
        cellfun(@delete_if_exists, armedPaths);
        cellfun(@delete_if_exists, donePaths);
        fprintf('[%d/%d] DONE sample %d\n', offset + 1, sampleCount, sampleId);
    end

    session.finishedUtc = utc_now();
    atomic_write_text(manifestPath, jsonencode(session, PrettyPrint=true));
    atomic_write_text(stopPath, sessionToken);
    wait_for_workers(futures, cfg.workerShutdownSeconds);
    fprintf('Acquisition complete: %s\n', outputRoot);
end


function cfg = make_config(outputRoot)
    cfg = struct();
    cfg.outputRoot = outputRoot;
    cfg.mmwave2dDir = fullfile(outputRoot, 'mm', '2d');
    cfg.mmwaveRaDir = fullfile(outputRoot, 'mm', 'ra');
    cfg.acousticDir = fullfile(outputRoot, 'ac');
    cfg.controlDir = fullfile(outputRoot, 'control');
    cfg.tokenPath = fullfile(cfg.controlDir, 'session_token.txt');
    cfg.pollSeconds = 0.005;
    cfg.workerTimeoutSeconds = 300;
    cfg.roundTimeoutSeconds = 120;
    cfg.workerShutdownSeconds = 30;
    cfg.acousticWarmupFrames = 100;
    cfg.acousticAverageFrames = 19;
    cfg.radarWarmupFrames = 10;
end


function create_directories(cfg)
    paths = {cfg.outputRoot, cfg.mmwave2dDir, cfg.mmwaveRaDir, ...
        cfg.acousticDir, cfg.controlDir};
    for index = 1:numel(paths)
        if ~exist(paths{index}, 'dir')
            mkdir(paths{index});
        end
    end
end


function clean_control_directory(controlDir)
    patterns = {'arm_*.json', 'go_*.txt', '*.armed', ...
        '*.done', '*_worker.ready', 'stop_*.txt', ...
        'session_token.txt'};
    for patternIndex = 1:numel(patterns)
        files = dir(fullfile(controlDir, patterns{patternIndex}));
        for fileIndex = 1:numel(files)
            delete_if_exists(fullfile(files(fileIndex).folder, files(fileIndex).name));
        end
    end
end


function acoustic_worker(cfg, sessionToken)
    stopPath = fullfile(cfg.controlDir, ['stop_' sessionToken '.txt']);
    Fs = 48e3;
    Fc = 20e3;
    sequenceLength = 480;
    zc = generate_zadoff_chu(59, 1);
    zcUp = interpft(zc, sequenceLength);
    time = (0:sequenceLength-1).' / Fs;
    probe = real(zcUp) .* cos(2*pi*Fc*time) ...
        - imag(zcUp) .* sin(2*pi*Fc*time);
    probe = 2 * probe(:);

    player = audioDeviceWriter('SampleRate', Fs);
    recorder = audioDeviceReader( ...
        'SampleRate', Fs, 'SamplesPerFrame', numel(probe));
    cleanup = onCleanup(@() release_audio(player, recorder));

    for index = 1:cfg.acousticWarmupFrames
        player(probe);
        recorder();
    end
    atomic_write_text(fullfile(cfg.controlDir, 'ac_worker.ready'), sessionToken);

    lastSampleId = 0;
    while ~isfile(stopPath)
        [sampleId, requestToken] = read_arm_request( ...
            cfg.controlDir, lastSampleId, sessionToken);
        if sampleId <= lastSampleId || ~strcmp(requestToken, sessionToken)
            pause(cfg.pollSeconds);
            continue;
        end

        expectedValue = sprintf('%s:%d', sessionToken, sampleId);
        atomic_write_text(fullfile(cfg.controlDir, ...
            sprintf('ac_%d.armed', sampleId)), expectedValue);
        if ~wait_for_go(cfg, sampleId, sessionToken, stopPath)
            return;
        end

        acquisitionStartedUtc = utc_now();
        cirSum = zeros(401, 1);
        for frameIndex = 1:cfg.acousticAverageFrames
            player(probe);
            received = recorder();
            alignedCir = acoustic_cir(received, zcUp, Fs, Fc);
            cirSum = cirSum + alignedCir(50:450);
        end
        energiess = abs(cirSum) / cfg.acousticAverageFrames;
        acquisitionFinishedUtc = utc_now();
        gateMetadata = struct( ...
            'protocol', 'ARM-BOTH_ARMED-SHARED_GO-BOTH_DONE', ...
            'sessionToken', sessionToken, ...
            'sampleId', sampleId, ...
            'acquisitionStartedUtc', acquisitionStartedUtc, ...
            'acquisitionFinishedUtc', acquisitionFinishedUtc, ...
            'averagedFrames', cfg.acousticAverageFrames);

        destination = fullfile(cfg.acousticDir, sprintf('%d.mat', sampleId));
        atomic_save_acoustic(destination, energiess, gateMetadata);
        atomic_write_text(fullfile(cfg.controlDir, ...
            sprintf('ac_%d.done', sampleId)), expectedValue);
        lastSampleId = sampleId;
    end
end


function mmwave_worker(cfg, sessionToken)
    stopPath = fullfile(cfg.controlDir, ['stop_' sessionToken '.txt']);
    radar = dca1000("IWR1843BOOST");
    cleanup = onCleanup(@() release_radar(radar));
    for index = 1:cfg.radarWarmupFrames
        radar();
    end
    atomic_write_text(fullfile(cfg.controlDir, 'mm_worker.ready'), sessionToken);

    lastSampleId = 0;
    while ~isfile(stopPath)
        [sampleId, requestToken] = read_arm_request( ...
            cfg.controlDir, lastSampleId, sessionToken);
        if sampleId <= lastSampleId || ~strcmp(requestToken, sessionToken)
            radar(); % Drain the DCA1000 UDP stream while waiting for ARM.
            continue;
        end

        expectedValue = sprintf('%s:%d', sessionToken, sampleId);
        atomic_write_text(fullfile(cfg.controlDir, ...
            sprintf('mm_%d.armed', sampleId)), expectedValue);
        if ~wait_for_go_with_radar(cfg, sampleId, sessionToken, stopPath, radar)
            return;
        end

        acquisitionStartedUtc = utc_now();
        rawData = radar();
        [b, doa_2d_db] = process_mmwave(rawData);
        acquisitionFinishedUtc = utc_now();
        gateMetadata = struct( ...
            'protocol', 'ARM-BOTH_ARMED-SHARED_GO-BOTH_DONE', ...
            'sessionToken', sessionToken, ...
            'sampleId', sampleId, ...
            'acquisitionStartedUtc', acquisitionStartedUtc, ...
            'acquisitionFinishedUtc', acquisitionFinishedUtc);

        destination2d = fullfile(cfg.mmwave2dDir, sprintf('%d.mat', sampleId));
        destinationRa = fullfile(cfg.mmwaveRaDir, sprintf('%d.mat', sampleId));
        atomic_save_mmwave(destination2d, destinationRa, ...
            b, doa_2d_db, gateMetadata);
        atomic_write_text(fullfile(cfg.controlDir, ...
            sprintf('mm_%d.done', sampleId)), expectedValue);
        lastSampleId = sampleId;
    end
end


function [sampleId, token] = read_arm_request(controlDir, lastId, sessionToken)
    sampleId = -1;
    token = '';
    files = dir(fullfile(controlDir, ...
        sprintf('arm_*_%s.json', sessionToken)));
    for index = 1:numel(files)
        try
            request = jsondecode(fileread( ...
                fullfile(files(index).folder, files(index).name)));
            candidateId = double(request.sampleId);
            candidateToken = char(request.sessionToken);
            phaseOk = ~isfield(request, 'phase') ...
                || strcmpi(char(request.phase), 'ARM');
            if phaseOk && isscalar(candidateId) && isfinite(candidateId) ...
                    && candidateId > lastId && strcmp(candidateToken, sessionToken)
                if sampleId < 0 || candidateId < sampleId
                    sampleId = candidateId;
                    token = candidateToken;
                end
            end
        catch
            % The coordinator may be atomically replacing the control file.
        end
    end
end


function ok = wait_for_go(cfg, sampleId, sessionToken, stopPath)
    ok = false;
    goPath = fullfile(cfg.controlDir, ...
        sprintf('go_%d_%s.txt', sampleId, sessionToken));
    expectedValue = sprintf('%s:%d', sessionToken, sampleId);
    while ~isfile(stopPath)
        if strcmp(read_text(goPath), expectedValue)
            ok = true;
            return;
        end
        pause(cfg.pollSeconds);
    end
end


function ok = wait_for_go_with_radar( ...
        cfg, sampleId, sessionToken, stopPath, radar)
    ok = false;
    goPath = fullfile(cfg.controlDir, ...
        sprintf('go_%d_%s.txt', sampleId, sessionToken));
    expectedValue = sprintf('%s:%d', sessionToken, sampleId);
    while ~isfile(stopPath)
        if strcmp(read_text(goPath), expectedValue)
            ok = true;
            return;
        end
        radar(); % Keep the UDP stream moving until the shared GO arrives.
    end
end


function alignedCir = acoustic_cir(received, zcUp, Fs, Fc)
    received = received(:);
    usableLength = floor(numel(received) / numel(zcUp)) * numel(zcUp);
    if usableLength < numel(zcUp)
        error('The recorded acoustic frame is too short.');
    end
    received = received(1:usableLength);
    sampleIndex = (0:usableLength-1).';
    baseband = received .* exp(-1j * 2*pi * Fc/Fs * sampleIndex);
    spectrum = fft(baseband);
    frequency = (0:usableLength-1).' * Fs / usableLength;
    keep = frequency <= 3.5e3 | frequency >= Fs - 3.5e3;
    spectrum(~keep) = 0;
    baseband = reshape(ifft(spectrum), numel(zcUp), []);
    cir = ifft(fft(baseband, [], 1) .* conj(fft(zcUp)), [], 1);
    [~, directIndex] = max(abs(cir(:, 1)));
    alignedCir = circshift(cir(:, 1), 1 - directIndex);
end


function sequence = generate_zadoff_chu(lengthValue, rootValue)
    index = (0:lengthValue-1).';
    if mod(lengthValue, 2) == 0
        sequence = exp(-1j*pi*rootValue*index.^2/lengthValue);
    else
        sequence = exp(-1j*pi*rootValue*index.*(index + 1)/lengthValue);
    end
end


function [b, rangeAzimuth] = process_mmwave(rawData)
    rangeBins = 256;
    dopplerBins = 128;
    angleBins = 128;
    txCount = 3;
    rawData = permute(rawData, [1, 3, 2]);
    virtualData = reorder_tdm_mimo(rawData, txCount);
    if size(virtualData, 3) < 12
        error('Expected twelve virtual mmWave channels.');
    end
    chirpCount = size(virtualData, 2);
    rangeWindow = hamming(size(virtualData, 1));
    dopplerWindow = hamming(chirpCount);
    rangeProfile = fft( ...
        virtualData .* reshape(rangeWindow, [], 1, 1), rangeBins, 1);
    speedProfile = fftshift(fft( ...
        rangeProfile .* reshape(dopplerWindow, 1, [], 1), ...
        dopplerBins, 2), 2);

    rangeEnergy = sum(abs(rangeProfile).^2, [2, 3]);
    roi = 7:20;
    [~, localIndex] = max(rangeEnergy(roi));
    peakBin = roi(localIndex);

    doa = zeros(angleBins);
    selectedBins = peakBin:min(peakBin + 1, rangeBins);
    for rangeIndex = selectedBins
        virtualAntennas = squeeze(speedProfile(rangeIndex, :, :)).';
        doa = doa + compute_2d_aoa(virtualAntennas, angleBins, dopplerBins);
    end
    b = flipud(doa.');

    rangeVirtual = permute(speedProfile(1:50, :, 1:8), [3, 2, 1]);
    padded = zeros(angleBins, dopplerBins, 50, 'like', speedProfile);
    padded(1:8, :, :) = rangeVirtual;
    spectrum = fft(padded, angleBins, 1);
    energy = squeeze(sum(abs(spectrum), 2));
    rangeAzimuth = fftshift(10 * log10(energy.^2 + 1e-6), 1).';
end


function virtualData = reorder_tdm_mimo(rawData, txCount)
    [sampleCount, chirpCount, receiverCount] = size(rawData);
    chirpsPerTx = floor(chirpCount / txCount);
    virtualData = zeros(sampleCount, chirpsPerTx, txCount * receiverCount, ...
        'like', rawData);
    for txIndex = 1:txCount
        chirpIndices = txIndex:txCount:chirpCount;
        chirpIndices = chirpIndices(1:chirpsPerTx);
        for receiverIndex = 1:receiverCount
            virtualIndex = (txIndex - 1) * receiverCount + receiverIndex;
            virtualData(:, :, virtualIndex) = ...
                rawData(:, chirpIndices, receiverIndex);
        end
    end
end


function doa = compute_2d_aoa(virtualAntennas, fftSize, dopplerBins)
    azimuth = zeros(fftSize, dopplerBins, 'like', virtualAntennas);
    azimuth(1:8, :) = virtualAntennas(1:8, :);
    azimuthSpectrum = fft(azimuth, fftSize, 1);
    elevation = zeros(fftSize, dopplerBins, 'like', virtualAntennas);
    elevation(1:2, :) = [virtualAntennas(3, :); virtualAntennas(9, :)];
    elevationSpectrum = fft(elevation, fftSize, 1);
    energy = abs(azimuthSpectrum) * abs(elevationSpectrum).';
    doa = fftshift(10 * log10(energy.^2 + 1e-6));
end


function atomic_save_acoustic(destination, energiess, gateMetadata)
    temporary = temporary_mat_path(destination);
    cleanup = onCleanup(@() delete_if_exists(temporary));
    save(temporary, 'energiess', 'gateMetadata');
    movefile(temporary, destination, 'f');
end


function atomic_save_mmwave( ...
        destination2d, destinationRa, b, doa_2d_db, gateMetadata)
    temporary2d = temporary_mat_path(destination2d);
    temporaryRa = temporary_mat_path(destinationRa);
    cleanup = onCleanup(@() delete_temporary_pair(temporary2d, temporaryRa));
    save(temporary2d, 'b', 'gateMetadata');
    save(temporaryRa, 'doa_2d_db', 'gateMetadata');
    movefile(temporary2d, destination2d, 'f');
    movefile(temporaryRa, destinationRa, 'f');
end


function path = temporary_mat_path(destination)
    [folder, name] = fileparts(destination);
    path = fullfile(folder, ...
        sprintf('.%s.%s.tmp.mat', name, char(java.util.UUID.randomUUID())));
end


function delete_temporary_pair(first, second)
    delete_if_exists(first);
    delete_if_exists(second);
end


function verify_sample_files(cfg, sampleId)
    paths = {
        fullfile(cfg.mmwave2dDir, sprintf('%d.mat', sampleId)), ...
        fullfile(cfg.mmwaveRaDir, sprintf('%d.mat', sampleId)), ...
        fullfile(cfg.acousticDir, sprintf('%d.mat', sampleId)) ...
    };
    for index = 1:numel(paths)
        if ~isfile(paths{index})
            error('Worker reported DONE but output is missing: %s', paths{index});
        end
    end
end


function wait_for_markers(paths, expectedValue, timeoutSeconds, futures)
    started = tic;
    while true
        ready = cellfun(@(path) strcmp(read_text(path), expectedValue), paths);
        if all(ready)
            return;
        end
        assert_workers_running(futures);
        if timeoutSeconds > 0 && toc(started) > timeoutSeconds
            error('Timed out waiting for gate markers: %s', strjoin(paths(~ready), ', '));
        end
        pause(0.005);
    end
end


function assert_workers_running(futures)
    for index = 1:numel(futures)
        future = futures(index);
        if strcmp(future.State, 'finished')
            if ~isempty(future.Error)
                rethrow(future.Error);
            end
            error('An acquisition worker exited before the session completed.');
        end
    end
end


function wait_for_workers(futures, timeoutSeconds)
    started = tic;
    while toc(started) <= timeoutSeconds
        if all(strcmp({futures.State}, 'finished'))
            for index = 1:numel(futures)
                if ~isempty(futures(index).Error)
                    rethrow(futures(index).Error);
                end
            end
            return;
        end
        pause(0.05);
    end
    for index = 1:numel(futures)
        if ~strcmp(futures(index).State, 'finished')
            cancel(futures(index));
        end
    end
end


function stop_workers(stopPath, futures)
    try
        atomic_write_text(stopPath, 'stop');
    catch
    end
    for index = 1:numel(futures)
        try
            if ~strcmp(futures(index).State, 'finished')
                cancel(futures(index));
            end
        catch
        end
    end
end


function release_audio(player, recorder)
    try
        if ~isempty(player), release(player); end
    catch
    end
    try
        if ~isempty(recorder), release(recorder); end
    catch
    end
end


function release_radar(radar)
    try
        if ~isempty(radar), radar.release(); end
    catch
    end
end


function value = read_text(path)
    value = '';
    try
        if isfile(path)
            value = strtrim(fileread(path));
        end
    catch
    end
end


function atomic_write_text(path, value)
    [folder, name, extension] = fileparts(path);
    if ~exist(folder, 'dir'), mkdir(folder); end
    temporary = fullfile(folder, sprintf('.%s.%s%s.tmp', ...
        name, char(java.util.UUID.randomUUID()), extension));
    cleanup = onCleanup(@() delete_if_exists(temporary));
    fileId = fopen(temporary, 'w', 'n', 'UTF-8');
    if fileId < 0
        error('Unable to create gate file: %s', temporary);
    end
    fileCleanup = onCleanup(@() fclose(fileId));
    fprintf(fileId, '%s', value);
    clear fileCleanup;
    movefile(temporary, path, 'f');
end


function delete_if_exists(path)
    try
        if isfile(path), delete(path); end
    catch
    end
end


function value = utc_now()
    value = char(datetime('now', 'TimeZone', 'UTC', ...
        'Format', 'yyyy-MM-dd''T''HH:mm:ss.SSSXXX'));
end
