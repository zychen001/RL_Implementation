% Problem:
% 1. Reproduce the Figure 9.1 in Example 9.1. Ignore the state distribution μ at the bottom,
% and ignore the distribution scale at the right side of the figure. To obtain the red
% curve for the true value vπ, use the policy evaluation method discussed in Chapter 4.1.
%
% 2. Generate the corresponding curve for semi-gradient TD(0) using the same setup as in Example
% 9.1, and then add the curve of semi-gradient TD(0) to the figure you generated for the first
% problem.

clear;
clc;
tic;

% Initialization
global params_MC;
global params_TD;
N_STATES = 1000;                           
STATES = 1:N_STATES;
START_STATE = 500;
END_STATES = [1, N_STATES+2];
ACTIONS = [-1, 1];
STEP_RANGE = 100;

ALPHA = 2e-5;
NUM_OF_GROUPS = 10;
NUM_OF_EPISODES = 200000;
group_size = N_STATES / NUM_OF_GROUPS;
params_MC = zeros(1, NUM_OF_GROUPS);
params_TD = zeros(1, NUM_OF_GROUPS);
% Compute true value function
true_value = compute_true_value(STATES, ACTIONS);


% Perform gradient Monte Carlo algorithm for specific episodes
for episode = 1:NUM_OF_EPISODES
    gradient_monte_carlo(ALPHA, START_STATE);
end

% Perform semi-gradient TD(0) algorithm for specific episodes
for episode = 1:NUM_OF_EPISODES
    semi_gradient_td(ALPHA, START_STATE);
end

% Plot results
x = 1:N_STATES;
y_true = true_value(2:N_STATES+1);
y_agg = zeros(1, N_STATES);
y_sgd = zeros(1, N_STATES);
for state = STATES
    y_agg(state) = value(state+1, params_MC);
    y_sgd(state) = value(state+1, params_TD);
end

figure;
hold on;
plot(x, y_true, 'r', 'LineWidth', 2);
plot(x, y_agg, 'b', 'LineWidth', 2);
plot(x, y_sgd, 'g', 'LineWidth', 2);
legend('True Value', 'Gradient Monte Carlo', 'Semi-gradient TD(0)');
xlabel('State');
ylabel('Value');
title('True Value vs Approximations');
toc;

% Functions to get constants
function END_STATES = get_END_STATES()
    END_STATES = [1, 1002];
end

function STEP_RANGE = get_STEP_RANGE()
    STEP_RANGE = 100;
end

function N_STATES = get_N_STATES()
    N_STATES = 1000;
end

% Get the value function estimation for a given state                                          
function val = value(state, params)
    if state == 1
        val = 0;
    elseif state == 1002
        val = 0;
    else
        group_index = floor((state - 1) / 100) + 1;
        group_index = min(group_index, 10);
        val = params(group_index);
    end
end

% Update the value function weights
function update(delta, state, tag)
    global params_MC;
    global params_TD;
    group_index = floor((state - 1) / 100) + 1;
    group_index = min(group_index, 10);
    if (tag == 1)
        params_MC(group_index) = params_MC(group_index) + delta;
    else
        params_TD(group_index) = params_TD(group_index) + delta;
    end
end

% Gradient Monte Carlo algorithm
function gradient_monte_carlo(alpha, START_STATE)
    END_STATES = get_END_STATES();
    global params_MC;
    
    state = START_STATE;
    trajectory = [state];
    reward = 0;
    
    while ~ismember(state, END_STATES)
        % Get an action
        a = action();
        % Get the next state and corresponding reward based on the action
        [next_state, r] = take_step(state, a);
        reward = r;
        % Record the visited states
        trajectory = [trajectory, next_state];
        state = next_state;
    end
    
    for i = 1:length(trajectory)-1
        state = trajectory(i);
        delta = alpha * (reward - value(state, params_MC));
        % Update the value function weights
        update(delta, state, 1);
    end
end

% Semi-gradient TD(0) algorithm
function semi_gradient_td(alpha, START_STATE)
    END_STATES = get_END_STATES();
    global params_TD;
    state = START_STATE;
    
    while ~ismember(state, END_STATES)
        % Get an action
        a = action();
        % Get the next state and corresponding reward based on the action
        [next_state, reward] = take_step(state, a);
        delta = reward + value(next_state, params_TD) - value(state, params_TD);
        % Update the value function weights
        update(alpha * delta, state, 2);
        state = next_state;
    end
end


function true_value = compute_true_value(STATES, ACTIONS)
    STEP_RANGE = get_STEP_RANGE();
    N_STATES = get_N_STATES();
    % An initial guess array of true value
    true_value = (-1001:2:1001) / 1001.0;

    % Compute the true state values using dynamic programming
    while true
        old_value = true_value;
        % For each states, updates the value by considering all posible
        % actions and the resulting next states
        for state = 2:1001
            true_value(state) = 0;
            for action = ACTIONS
                for step = 1:STEP_RANGE
                    step = step * action;
                    next_state = state + step;
                    next_state = max(min(next_state, N_STATES + 2), 1);
                    % Asynchronous update for faster convergence
                    true_value(state) = true_value(state) + 1.0 / (2 * STEP_RANGE) * true_value(next_state);
                end
            end
        end
        error = sum(abs(old_value - true_value));
        % Ends the loop
        if error < 1e-2
            break;
        end
    end
    % Correct the state value for terminal states
    true_value(1) = -1;
    true_value(end) = 1;
end



% Taking a step in the Random Walk enviroment with the picked action
function [state, reward] = take_step(state, action)
    STEP_RANGE = get_STEP_RANGE();
    N_STATES = get_N_STATES();
    step = randi([1, STEP_RANGE+1]) * action;
    state = state + step;
    state = max(min(state, N_STATES + 2), 1);
    
    if state == 1
        reward = -1;
    elseif state == N_STATES + 2
        reward = 1;
    else
        reward = 0;
    end
end

% Get an action following a random policy
function a = action()
    a = 0;
    if rand() < 0.5
        a = -1;
    else
        a = 1;
    end
end






   
