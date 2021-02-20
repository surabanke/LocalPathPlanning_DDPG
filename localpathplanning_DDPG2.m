%% Bus Creation
% Create the bus of actors from the scenario reader
ModelName = 'new_lpp_sim_matfile'; % simulink model name
wasModelLoaded = bdIsLoaded(ModelName);
if ~wasModelLoaded
    load_system(ModelName)
end
blk=find_system(ModelName,'System','driving.scenario.internal.ScenarioReader');
s = get_param(blk{1},'PortHandles');
get(s.Outport(1),'SignalHierarchy');

[scenario,egoCar,actor_profiles] = helperSessionToScenario('temp_lpp_env.mat');

ego_x = egoCar.x0;
ego_y = egoCar.y0;
ego_v = egoCar.v0;
ego_yaw = egoCar.yaw0;
m = 1575 ; %vehivle mass(kg)
lf = 1.2;
lr = 1.6;
lz = 2875;
%% open simulink and define state, action, env

open_system(ModelName)
agent_Block = [ModelName, '/RL Agent'];
Tf = 10 ; % seconds, simulation duration
Ts =  0.1; % seconds (15ms), simulation time per one step


ObservationInfo = rlNumericSpec([9 3]);
ObservationInfo.Name = 'ENV_States';


ActionInfo = rlNumericSpec([3 1], ...
    'LowerLimit',[ 0; 0; -180],...
    'UpperLimit',[ 10; 10; 180]);


ActionInfo.Name = 'ENV_Action';

env = rlSimulinkEnv(ModelName,agent_Block, ObservationInfo, ActionInfo);
%env.ResetFcn = @(in) LocalPathPlanningFcn(in);
%env.UseFastRestart = 'off';
rng(0);


%% DDPG critic Network

statePath = [
    imageInputLayer([9 3 1],'Normalization','none','Name','state')
    
    fullyConnectedLayer(54,'Name','CriticStateFC1')
    reluLayer('Name','CriticRelu1')
    fullyConnectedLayer(27,'Name','CriticStateFC2')
    reluLayer('Name','CriticRelu2')
    fullyConnectedLayer(18,'Name','CriticStateFC3')
    additionLayer(2,'Name','add')
    reluLayer('Name','CriticRelu3')
    fullyConnectedLayer(1,'Name','CriticStateFC4')];
    
  

actionPath = [
    imageInputLayer([3 1 1],'Normalization','none','Name','action')
    fullyConnectedLayer(18,'Name','CriticActionFC1')]; % by the number of action

criticNetwork = layerGraph(statePath);
criticNetwork = addLayers(criticNetwork, actionPath);
criticNetwork = connectLayers(criticNetwork,'CriticActionFC1','add/in2');


figure
plot(criticNetwork)


criticOpts = rlRepresentationOptions('LearnRate',1e-3,'GradientThreshold',1);
critic = rlQValueRepresentation(criticNetwork,ObservationInfo,ActionInfo,'Observation',{'state'},'Action',{'action'},criticOpts);

%% Actor network
actorNetwork = [
    imageInputLayer([9 3 1],'Normalization','none','Name','state')
    
    fullyConnectedLayer(54,'Name','CriticStateFC1')
    reluLayer('Name','CriticRelu1')
    fullyConnectedLayer(27,'Name','CriticStateFC2')
    reluLayer('Name','CriticRelu2')
    fullyConnectedLayer(3,'Name','CriticStateFC3')
    
    tanhLayer('Name','tanh1')];

actorOpts = rlRepresentationOptions('LearnRate',1e-4,'GradientThreshold',1); 

actor = rlDeterministicActorRepresentation(actorNetwork,ObservationInfo,ActionInfo,...
    'Observation',{'state'},'Action',{'tanh1'},actorOpts);

%% define agent 
agentOptions = rlDDPGAgentOptions(...
    'SampleTime',Ts,...
    'TargetSmoothFactor',1e-3,...
    'ExperienceBufferLength',1e6 ,...
    'DiscountFactor',0.99,...
    'MiniBatchSize',256);  

%agentOptions.NoiseOptions.Variance = 1e-1;
%agentOptions.NoiseOptions.VarianceDecayRate = 1e-6; 
%agentOptions.SaveExperienceBufferWithAgent = true; % Default false
%agentOptions.ResetExperienceBufferBeforeTraining = false; % Default true

agent = rlDDPGAgent(actor,critic,agentOptions);


%% Train DDQN

maxsteps = ceil(Tf/Ts); 
trainingOptions = rlTrainingOptions(...
    'MaxEpisodes',10000,...
    'MaxStepsPerEpisode',maxsteps,...
    'StopOnError',"on",...
    'Verbose',false,...
    'Plots',"training-progress",...
    'StopTrainingCriteria',"AverageReward",...
    'StopTrainingValue',45,...
    'ScoreAveragingWindowLength',20,...
    'SaveAgentCriteria',"AverageReward",...
    'SaveAgentValue',45); 

%trainingOptions.UseParallel = true;
%trainingOptions.ParallelizationOptions.Mode = "async";
%trainingOptions.ParallelizationOptions.DataToSendFromWorkers = "experiences";
%trainingOptions.ParallelizationOptions.StepsUntilDataIsSent = 24;


%% Train the agent.
doTraining = true;

if doTraining
    trainingStats = train(agent,env,trainingOptions);
end
save(trainingOptions.SaveAgentDirectory + "finalAgent.mat",'agent')

%% Simulate trained Agent

%simOptions = rlSimulationOptions('MaxSteps',2000);
%experience = sim(env,agent,simOptions);

%totalReward = sum(experience.Reward);


