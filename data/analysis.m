%% Analysis filter results
% Depending on the number of particles you are testing
% you will need to hardcode them in the particles variable

function analysis()
    
    unit = ["CPU", "GPU"];
    particles = ["65536", "1048576"];
    methods = ["Systematic", "Stratified", "MetropolisC2"];
    header = ('%f %f %f %f');
    
    dir = ('./');
    
    for u = 1:length(unit)
        if (u == 1)
            disp('Check CPU')
        else
            disp('Check GPU')
        end
        for p = 1:length(particles)
            for m = 1:length(methods)
                
                try
                    %% Read input
                    filename = strcat(dir, 'truth_', unit{u}, '_', methods{m}, '_', num2str(particles(p)), '.txt');
                    
                    % open the file
                    fid = fopen(filename);
                    
                    % read header
                    C = textscan(fid, header, 1);
                    fgetl(fid);
                    MC = C{1};
                    states = C{2};
                    samples = C{3};
                    
                    s.meas = zeros(samples, states);
                    inputStruct = repmat(s, MC, 1);
                    
                    for n = 1:MC
                        inputStruct(n).meas = fscanf(fid, '%f ', [states samples])';
                    end
                    
                    fclose(fid);
                    
                    %% Read BPF GPU
                    filename = strcat(dir, 'estimate_', unit{u}, '_', methods{m}, '_', num2str(particles(p)), '.txt');
                    
                    % open the file
                    fid = fopen(filename);
                    
                    % read header
                    C = textscan(fid, header, 1);
                    fgetl(fid);
                    
                    MC = C{1};
                    states = C{2};
                    samples = C{3};
                    
                    % initialize struct
                    s.meas = zeros(samples, states);
                    outputStruct = repmat(s, MC, 1);
                    
                    for n = 1:MC
                        fgetl(fid); % Skip timing information
                        outputStruct(n).meas = fscanf(fid, '%f ', [states samples])';
                    end
                    
                    fclose(fid);
                    
                    RMSE_PF  = zeros(1,4);
                    
                    for n = 1:MC
                        estErrorPF = outputStruct(n).meas - inputStruct(n).meas;
                        
                        RMSE_PF = RMSE_PF + sum(estErrorPF.^2);
                    end
                    RMSE_PF  = sqrt((1/(samples*MC))*RMSE_PF);
                    
                    disp(strcat ({'RMSE for '}, num2str(particles(p)), {' for '}, methods{m}))
                    disp(RMSE_PF)
                catch
                    disp(strcat ({'No file '}, filename));
                end
            end
        end
    end
end