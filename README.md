# LocalPathPlanning_DDPG
ego car learning Local path planning with DDPG agent.



MATLAB R2020b
- Automated driving toolbox (DrivingScenarioDesigner - Export it as a matlab function - make Scenario1 at workspace)
- Reinforcement learning toolbox


localpathplanning_DDPG2.m 이 메인코드
new_lpp_sim_matfile.slx 는 Simulink (observation 상태정보, reward, episode is Done 을 알수 있는 정보를 구하는 코드들이 들어있다.)
savedAgentsfinalAgent 는 학습이 모두 끝나면 나오는 가장 잘 학습 되었다고 여겨지는 에이전트(딥러닝에선 마지막 웨이트 파일 같은 존재)
savedAgents 45점의 리워드를 넘어야 저장됨
busActor1는 학습하기 전에 항상 workspace에 존재하는지 확인
temp_lpp_env는 drivingScenarioDesigner파일로 들어가서 환경을 변경해 줄 수 있음.
