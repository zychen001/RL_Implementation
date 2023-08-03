% Problem:
% Consider the same game as in coin_toss.
%  • Write Matlab code to implement the policy iteration algorithm and find the optimal policy
%    when the probability of head p = 0.9.
%  • Write Matlab code to implement the value iteration algorithm and find the optimal policy
%    when the probability of head p = 0.1.



clc;
close all;
clear;
tic;
gamma = 1;
% 0 and 10 are terminal states
n_non_term_states = 9;
n_all_states = n_non_term_states + 2;
% ---------------- policy iteration ------------------------
%  initialization 
%  V: state-value function, pi: policy
V = zeros(1, n_all_states); 
pi = zeros(1, n_all_states);

policy_stable = 0;
p_head = 0.9;
theta = 1e-8;
% once the policy is stable, stop the iteration
while ( policy_stable == 0)
    % policy evaluation
    V = evaluation_policy_iteration(gamma, n_all_states, theta, V, pi, p_head);
    % policy improvement
    [V, pi, policy_stable] = policy_improvement(V, pi, gamma, n_all_states, p_head, policy_stable);
end
fprintf("policy iteration, probability of Head: 0.9\n")

fprintf("V(s):\n")
for i = 1 : 11
    idx = i - 1;
    fprintf("State" + idx + " : ");
    fprintf(V(i) + "\n");
end
fprintf("\n");
fprintf("π(s):\n")
for i = 1 : 11
    idx = i - 1;
    fprintf("State" + idx + " : ");
    fprintf(pi(i) + "\n");
end
fprintf("\n");

% ----------------- value iteration ------------------------
% initialization
V = zeros(1, n_all_states); 
p_head = 0.1;
pi = zeros(1, n_all_states);
% policy evaluation for value iteration
V = evaluation_value_iteration(gamma, n_all_states, theta, V, pi, p_head);
% get the deterministic policy for each state
[V, pi] = policy_improvement(V, pi, gamma, n_all_states, p_head, policy_stable);

fprintf("value iteration, probability of Head: 0.1\n")
fprintf("V(s):\n")
for i = 1 : 11
    idx = i - 1;
    fprintf("State" + idx + " : ");
    fprintf(V(i) + "\n");
end
fprintf("\n");
fprintf("π(s):\n")
for i = 1 : 11
    idx = i - 1;
    fprintf("State" + idx + " : ");
    fprintf(pi(i) + "\n");
end
fprintf("\n");
toc;

function [V, pi, policy_stable] = policy_improvement(V, pi, gamma, n_states, p_head, policy_stable)
    
    % assume the policy is stable (only use this in policy iteration)
    policy_stable = 1;
    for s = 2 : n_states-1
        a = pi(s);
        % find the action that maximize Q
        max_val = -Inf;
        max_act = 0;
        for act = 1 : s-1
            Q = state_bellman(s, act, V, gamma, p_head);
            if (Q > max_val)
                max_val = Q;
                max_act = act;
            end
        end
        % get the improved action in this state
        pi(s) = max_act;
        if (pi(s) ~= a)
            policy_stable = 0;
        end
    end

end


function [V] = evaluation_policy_iteration(gamma, n_states, theta, V, pi, p_head)
    delta = +Inf;
    while (delta > theta) 
        delta = 0;
        % loop over all non_terminal states
        for s = 2 : n_states-1
            v = V(s);
            % update the state value function
            V(s) = state_bellman(s, pi(s), V, gamma, p_head);
            % update delta to be the maximum of two value
            delta = max(delta, abs(v - V(s)));
        end
    end
end

function [V] = evaluation_value_iteration(gamma, n_states, theta, V, pi, p_head)
    delta = +Inf;
    while (delta > theta) 
        delta = 0;
        % loop over all non_terminal states
        for s = 2 : n_states-1
            v = V(s);
            % try all actions through bellman function, get the maximum Q
            Q = zeros(1, s-1);
            for act = 1 : s-1
                Q(act) = state_bellman(s, act, V, gamma, p_head);
            end
            V(s) = max(Q);
            % update delta to be the maximum of two value
            delta = max(delta, abs(v - V(s)));
        end
    end
end

function [v] = state_bellman(s,bet,V,gamma,p_head)
    
    s_head = s + bet; % coin toss is head, the player win the bet
    s_tail = s - bet; % coin toss is tail, the player lose the bet
    v = 0;

    if( s_head > 10 ) 
      % if the player's money is over 10 dollars, it reaches terminal state 
      s_head = 11; 
    end                
    v = p_head*( bet + gamma*V(s_head) ); 


    if( s_tail <= 1 )  
      % if the player's money is lower 0 dollars, it reaches terminal state
      s_tail = 1;
    end              
    v = v + (1-p_head)*( -bet + gamma*V(s_tail) ); 

end