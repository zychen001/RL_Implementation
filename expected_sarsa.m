% Problem:
% 1. Reproduce the Figure in Example 6.6 of the book. You don’t need to plot the figure for the
% Gridworld, just plot the figure showing the performance (i.e., the figure where the x-axis is
% the episodes and y-axis is sum of rewards during episode)
%
% 2. Generate the curve for Expected Sarsa using the same setup in Example 6.6, and then add
% the curve for Expected Sarsa to the figure you generated for the first problem (i.e., you will
% have three curves: Sarsa, Q-learning and Expected Sarsa)


close all;
clear all;
clc;
tic;

% Initialization
% Height and width of the grid
global HEIGHT; HEIGHT = 4;                  % height
global WIDTH; WIDTH = 12;                   % width
% The starting position and goal position of the grid
START_STATE = [4, 1];   %Start position
GOAL_STATE = [4, 12];   %Goal position
% There are 4 action: go left, go right, go up, go down
N_ACTIONS = 4;
global LEFT; global RIGHT; global UP; global DOWN;
LEFT = 1; RIGHT = 2; UP = 3; DOWN = 4;


ALPHA = 0.5;                                % Step size for Q-Learning and Sarsa
GAMMA = 1;                                  % GAMMA is 1 for Undiscounted, episodic task
N_episodes = 500;                           % Number of episodes
N_runs = 50;                               % Number of runs
epsilon = 0.1;                              % Probability of exploration in e_greedy method

% implement Expected SARSA with e-greedy 
avg_sum_rewards = zeros(1, N_episodes);
for i = 1:N_runs
    Q = zeros(WIDTH*HEIGHT, N_ACTIONS);
    for j = 1:N_episodes
        [sum_rewards, Q] = expected_sarsa(Q, ALPHA, GAMMA, epsilon, START_STATE, GOAL_STATE, HEIGHT, N_ACTIONS);
        avg_sum_rewards(j) = avg_sum_rewards(j) + sum_rewards;
    end
end
avg_sum_rewards = avg_sum_rewards/N_runs;   
expected_sarsa_result = avg_sum_rewards; 
expected_sarsa_result = movmean(expected_sarsa_result, 10);

% implement SARSA with e-greedy
avg_sum_rewards = zeros(1, N_episodes);
for i = 1:N_runs
    Q = zeros(WIDTH*HEIGHT, N_ACTIONS);
    for j = 1:N_episodes
        [sum_rewards, Q] = sarsa(Q, ALPHA, GAMMA, epsilon, START_STATE, GOAL_STATE, HEIGHT);
        avg_sum_rewards(j) = avg_sum_rewards(j) + sum_rewards;
    end
end
avg_sum_rewards = avg_sum_rewards/N_runs;   
sarsa_result = avg_sum_rewards;  
sarsa_result = movmean(sarsa_result, 10);

% implement Q-learning with e-greedy

avg_sum_rewards = zeros(1, N_episodes);
for i = 1:N_runs
    Q = zeros(WIDTH*HEIGHT, N_ACTIONS);
    for j = 1:N_episodes
        [sum_rewards, Q] = qlearning(Q, ALPHA, GAMMA, epsilon, START_STATE, GOAL_STATE, HEIGHT);
        avg_sum_rewards(j) = avg_sum_rewards(j) + sum_rewards;
    end
end
avg_sum_rewards = avg_sum_rewards/N_runs;   
qlearning_result = avg_sum_rewards;
qlearning_result = movmean(qlearning_result, 10);

% plot 
figure(1), title('SARSA vs Q-learning vs Expected SARSA(e-greedy policy, e=0.1)'), xlabel('Episodes'), ylabel('Sum of rewards during episode'), legend(), xlim([0 500]), ylim([-100 0]), hold on, 
plot(expected_sarsa_result, 'g', 'DisplayName', 'Expected Sarsa')
plot(sarsa_result, 'b', 'DisplayName', 'SARSA'),
plot(qlearning_result, 'r', 'DisplayName', 'Q-learning')
toc;



% Expected SARSA with e-greedy policy function 
function [reward_sum, Q_new] = expected_sarsa(Q, alpha, gamma, epsilon, START_STATE, GOAL_STATE, HEIGHT, N_ACTIONS)

    State = START_STATE;
    reward_sum = 0;
    while(~isequal(State, GOAL_STATE))
        % get an action using the epsilon-greedy policy
        A = e_greedy(epsilon, State, Q, HEIGHT);
        % get the next state and reward
        [State_next, R] = take_action(State, A, START_STATE);
        reward_sum = reward_sum + R;
        
        % Calculate the expected value of Q(S', A')
        Q_next_state = Q((State_next(2)-1)*HEIGHT+State_next(1), :);
        policy_probs = epsilon/N_ACTIONS * ones(1, N_ACTIONS); % probability distributions
        policy_probs(find(Q_next_state == max(Q_next_state))) = (1 - epsilon) + epsilon/N_ACTIONS;
        Q_next = sum(Q_next_state .* policy_probs);
        
        % Update Q(S, A)
        S = (State(2)-1)*HEIGHT+State(1);
        Q(S, A) = Q(S, A) + alpha * (R + gamma * Q_next - Q(S, A));
        
        State = State_next;
    end
    % return and update the Q table
    Q_new = Q;
end

% SARSA with e-greedy policy function
function [reward_sum, Q_new] = sarsa(Q, alpha, gamma, epsilon, START_STATE, GOAL_STATE, HEIGHT)

    State = START_STATE;
    reward_sum = 0;
    % get an action using the epsilon-greedy policy
    A = e_greedy(epsilon, State, Q, HEIGHT);
    while(~isequal(State, GOAL_STATE))
        % get the next state and reward
        [State_next, R] = take_action(State, A, START_STATE);
        A_next = e_greedy(epsilon, State_next, Q, HEIGHT);
        reward_sum = reward_sum + R;
        %update Q(S, A)
        S = (State(2)-1)*HEIGHT+State(1);
        S_next = (State_next(2)-1)*HEIGHT+State_next(1);
        Q(S,A) = Q(S,A) + alpha*(R + gamma*Q(S_next,A_next) - Q(S,A));
        State = State_next;
        A = A_next;
    end
    % return and update the Q table
    Q_new = Q;
end

% Q-learning with e-greedy policy function
function [reward_sum, Q_new] = qlearning(Q, alpha, gamma, epsilon, START_STATE, GOAL_STATE, HEIGHT)

    State = START_STATE;
    reward_sum = 0;
    while(~isequal(State, GOAL_STATE))
        % get an action using the epsilon-greedy policy
        A = e_greedy(epsilon, State, Q, HEIGHT);
        % get the next state and reward
        [State_next, R] = take_action(State, A, START_STATE);
        reward_sum = reward_sum + R;
        %update Q(S, A)
        S = (State(2)-1)*HEIGHT+State(1);
        S_next = (State_next(2)-1)*HEIGHT+State_next(1);
        Q(S,A) = Q(S,A) + alpha*(R + gamma*max(Q(S_next,:)) - Q(S,A));
        State = State_next;
    end
    % return and update the Q table
    Q_new = Q;
end

%%% e-greedy policy function %%%
function A = e_greedy(epsilon, State, Q, HEIGHT)

    Q_state_all_actions = Q((State(2)-1)*HEIGHT+State(1),:);
    [~, greedy_action]= max(Q_state_all_actions);
    p = randi(length(greedy_action)); 
    A_greedy = greedy_action(p);
    A_explore = randi(length(Q_state_all_actions));  
    
    x = rand;
    if(x < epsilon)
        % Choose a random action with probability epsilon
        A = A_explore;
    else            
        % Choose the action with the highest Q-value for the current state
        A = A_greedy;
    end
end

%%% Takes an action in the current state %%%
function [State_next, R] = take_action(State, A, START_STATE)

    % Define the possible actions
    global LEFT; global RIGHT; global UP; global DOWN;
    global HEIGHT; global WIDTH;
    
    row = State(1); col = State(2);
    % Determine the next state based on the action taken and make sure it
    % is within the boundaries
    if(A == LEFT)
        State_next = [row, max(col-1, 1)];
    elseif(A == RIGHT)
        State_next = [row, min(col+1, WIDTH)];
    elseif(A == UP)
        State_next = [max(row-1, 1), col];
    else
        State_next = [min(row+1, HEIGHT), col];
    end
    
    R = -1;
    % if the agent hit the cliff, it gain a reward of -100 and return to 
    % the starting state, otherwise the reward is -1 
    if(State_next(1) == 4 && State_next(2) >= 2 && State_next(2) <= 11)
        R = -100;
        State_next = START_STATE;
    end
end
