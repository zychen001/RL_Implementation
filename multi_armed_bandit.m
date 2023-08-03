% Problem: Write Matlab program to implement -greedy algorithm and UCB algorithm to reproduce Fig. 2.4 
% of the textbook.


clc;
close all;
tic;
% initialization
avgr1 = zeros(1,1000);
avgr2 = zeros(1,1000);
avgr_ucb = zeros(1,1000);
avgr_egreedy = zeros(1,1000);

% 2000 runs
for r = 1:2000
    % set the initial values for 10-armed test
    q = randn(1,10);
    Q1 = zeros(1,10); % estimations
    Q2 = zeros(1,10);
    N1 = zeros(1,10); % number of times each action has been selected
    N2 = zeros(1,10);
       
    % ucb method
    avgr1=ucb(Q1,q,N1,avgr1);
    avgr_ucb = avgr1 / r;
    
    % epsilon-greedy method
    avgr2=egreedy(Q2,q,N2,avgr2);
    avgr_egreedy=avgr2/r;
end

plot(avgr_ucb,'b');
hold on;
plot(avgr_egreedy,'red');
legend('ucb', 'ε-greedy');
ylabel('Average Reward');
xlabel('Steps');
toc;

function [avgr2]=egreedy(Q,q,N,avgr2)
     % 1000 time steps per run
     for t = 1:1000
        % if tag == 0, explore
        tag = 0;
        px = rand;
        if px < 0.9
            tag=1;
        end

        m = max(Q);
        maxindex = find(Q == m);
        % select action
        if tag == 1
            if length(maxindex) ~= 1
                action_idx = maxindex(ceil(rand()*length(maxindex)));
            else
                action_idx = maxindex(1);
            end
            N(action_idx) = N(action_idx)+1;
            R = normrnd(q(action_idx),1);
            % update estimation
            Q(action_idx) = Q(action_idx) + (R-Q(action_idx))/N(action_idx);
            avgr2(t) = avgr2(t) + R; 
        else
            action_idx = randi(10);
            N(action_idx) = N(action_idx)+1;
            R = normrnd(q(action_idx),1);
            % update estimation
            Q(action_idx) = Q(action_idx) + (R-Q(action_idx))/N(action_idx);
            % record average reward
            avgr2(t) = avgr2(t) + R; 
        end
        clear maxindex
    end
end

function [avgr1]=ucb(Q,q,N,avgr1)
     c = 2;
     for t = 1:1000
        
        UCB = Q + c*sqrt(log(t)./N);
        [~, maxindex] = max(UCB);

        
        if length(maxindex) ~= 1
            action_index = maxindex(ceil(rand()*length(maxindex)));
        else
            action_index = maxindex(1);
        end
        N(action_index) = N(action_index)+1;
        r = normrnd(q(action_index),1);
        % update estimation
        Q(action_index) = Q(action_index)+(r-Q(action_index))/N(action_index);
        % record average reward
        avgr1(t) = avgr1(t) + r; 
        clear maxindex
    end
end
