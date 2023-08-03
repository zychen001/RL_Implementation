% Problem:
% Consider the game explained in the class. You start with 5 dollars. At each time step, you can
% decide how much to bet. After you put your bet, a coin will be tossed. If the result of the coin toss
% is Head, you win the amount of money you bet. If the result of the coin toss is Tail, you lose the
% money you bet. The game ends once you have no money left or you have 10 or more dollars. Your
% goal is to win the maximum total amount. Set γ = 1 as this is an episodic task.

% 1. Assume the probability of Head is 0.9. Write matlab program to implement the policy eval-
% uation algorithm, and compute vπ(s) for the following policies:
% • The aggressive policy, in which you always bet the maximum amount allowed. For
% example, if you have 7 dollars, you bet 7 dollars. If you have 3 dollars, you bet 3 dollars.
% • The conservative policy, in which you always bet 1 dollar no matter how much money
% you have.
% • The random policy, in which you randomly pick an amount to bet with uniform distri-
% bution. For example, if you have 3 dollars left, you will randomly pick a number from
% {1, 2, 3} to bet, each of which has probability 1/3. As another example, if you have 8
% dollars, you will randomly pick a number from {1, 2, 3, · · · , 8} to bet, each of which has
% probability 1/8.

% 2. Now assume the probability of Head is 0.1. Repeat the same questions above


clc;
close all;
tic;
gamma = 1;
% 0 and 10 are terminal states
n_non_term_states = 9;
n_all_states = n_non_term_states + 2;

% initialize the state value function
V_aggressive = zeros(1, n_all_states); 
V_aggressive(1)=0.0; 
V_aggressive(6)=5.0; 
V_aggressive(end)=0.0;

V_conservative = zeros(1, n_all_states); 
V_conservative(1)=0.0; 
V_conservative(6)=5.0; 
V_conservative(end)=0.0;

V_random = zeros(1, n_all_states); 
V_random(1)=0.0; 
V_random(6)=5.0; 
V_random(end)=0.0;

p_head = 0.9;
theta = 1e-8;
% policy evaluation when the probability of Head is 0.9
eval(gamma, n_all_states, theta, V_aggressive, V_conservative, V_random, p_head);



% initialize the state value function
V_aggressive = zeros(1, n_all_states); 
V_aggressive(1)=0.0; 
V_aggressive(6)=5.0; 
V_aggressive(end)=0.0;

V_conservative = zeros(1, n_all_states); 
V_conservative(1)=0.0; 
V_conservative(6)=5.0; 
V_conservative(end)=0.0;

V_random = zeros(1, n_all_states); 
V_random(1)=0.0; 
V_random(6)=5.0; 
V_random(end)=0.0;

p_head = 0.1;
theta = 1e-8;
% policy evaluation when the probability of Head is 0.1
eval(gamma, n_all_states, theta, V_aggressive, V_conservative, V_random, p_head);

toc;


function [] = eval(gamma, n_states, theta, V_aggressive, V_conservative, V_random, p_head)
    delta = +Inf;
    while (delta > theta) 
        delta = 0;
        % loop over all non_terminal states
        for sidx = 2 : n_states-1
            v = V_aggressive(sidx);
            % s is the current money of player
            s = sidx - 1;
            bet = s;
            
            Q = state_bellman(s, bet, V_aggressive, gamma, p_head);
            % update the state value function
            V_aggressive(sidx) = Q;
            % update delta to be the maximum of two value
            delta = max(delta, abs(v - V_aggressive(sidx)));
        end
    end

    
    
    
    delta = +Inf;
    while (delta > theta) 
        delta = 0;
        % loop over all non_terminal states
        for sidx = 2 : n_states-1
            v = V_conservative(sidx);
            % s is the current money of player
            s = sidx - 1;
            bet = 1;
            Q = state_bellman(s, bet, V_conservative, gamma, p_head);
            % update the state value function
            V_conservative(sidx) = Q;
            % update delta to be the maximum of two value
            delta = max(delta, abs(v - V_conservative(sidx)));
        end
    end

    
    
    
    delta = +Inf;
    iterCnts = 0;
    while (delta > theta) 
        iterCnts = iterCnts + 1;
        delta = 0;
        % loop over all non_terminal states
        for sidx = 2 : n_states-1
            v = V_random(sidx);
            % s is the current money of player
            s = sidx - 1;
            bet = randi(s);
    
            Q = state_bellman(s, bet, V_random, gamma, p_head);
            % update the state value function
            if (V_random(sidx) < Q)
                V_random(sidx) = Q;
            end
            % update delta to be the maximum of two value
            delta = max(delta, abs(v - V_random(sidx)));
        end
    end

    fprintf("Probability of Head: " + p_head + "\n");
    fprintf("vπ(s) for aggressive policy:\n")
    for i = 1 : 11
        idx = i - 1;
        fprintf("State" + idx + " : ");
        fprintf(V_aggressive(i) + "\n");
    end
    fprintf("vπ(s) for conservative policy:\n")
    for i = 1 : 11
        idx = i - 1;
        fprintf("State" + idx + " : ");
        fprintf(V_conservative(i) + "\n");
    end
    fprintf("vπ(s) for random policy:\n")
    for i = 1 : 11
        idx = i - 1;
        fprintf("State" + idx + " : ");
        fprintf(V_random(i) + "\n");
    end
    fprintf("\n");
end

function [v] = state_bellman(s,bet,V,gamma,p_head)

    s_head = s + bet; % coin toss is head, the player win the bet
    s_tail = s - bet; % coin toss is tail, the player lose the bet
  

    v = 0;
    
    if( s_head > 10 ) 
      % if the player's money is more than 10 dollars, it reach state10
      bet = 10; 
      s_head = 10;
      % v = p_head*( bet + gamma*V(s_head+1));  
    end                
    v = p_head*( bet + gamma*V(s_head+1) ); 
    
    
    if( s_tail <= 0 )  
      % if the player's money is less than 0 dollars, it reach state1
      s_tail = 0; v = v + (1-p_head)* (-bet);
    else              
      v = v + (1-p_head)*( -bet + gamma*V(s_tail+1) ); 
    end

end