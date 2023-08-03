% Problem:
% Reproduce Figure 5.2 using Monte Carlo ES.
% A “Soft 17” means an Ace and 6. The book is not clear what is the dealer’s behavior in “Soft
% 17”. In your program, please consider two separate cases: 
%
% 1) Case 1, the dealer must hit in “Soft17”; 
% 2) Case 2, the dealer must stick in “Soft 17”. In your report, please include figures generated
% by these two cases separately.
%
% If the dealer has ‘soft 18” (Ace+7) or “Soft 19” (Ace+8) or “Soft 20” (Ace+9), the dealer must
% stick.

close all;
clear;
clc;
tic;

episodes = 5e6;


currentSum_states = 21 - 12 + 1;
card_showing_states = 13;
action_states = 2;
% the total states in one episode
total_states = currentSum_states * card_showing_states * action_states;
% policy array. stick - "0", hit - "1", the initial policy will be always hit
policy = ones(1, total_states);
% action-value function matrix
Q = zeros(total_states, action_states);
% returns array to record the reward
returns = zeros(total_states, action_states);
returns_cnt = zeros(total_states, action_states);

for i = 1:episodes
    % record the states in the current episode
    state_current = [];
    % shuffle the cards and get a new deck
    % There are 52 cards in total. 
    % 1, 2, 3, ..., J, Q, K for diamonds, clubs, heart and spades
    % the face cards will be count as 10
    deck = randperm(52);
    

    % The game begins with two cards dealt to both player and dealer
    % p_cards: the player's current cards 
    % p_sum: the numerical sum of player's cards
    p_cards = deck(1:2);
    deck = deck(3:end);
    [p_sum, usable_ace] = calculate_sum(p_cards);
    % d_cards: the dealer's current cards 
    % d_sum: the numerical sum of dealer's cards
    d_cards = deck(1:2);
    deck = deck(3:end);
    d_sum = calculate_sum(d_cards);
    card_showing = d_cards(1); % one of the dealer's card is face up
     
    % only records the states that card sum is equal or over 12
    while (p_sum < 12)
        p_cards = [ p_cards, deck(1) ]; 
        deck = deck(2:end); 
        [p_sum, usable_ace] = calculate_sum(p_cards); 
    end

    % store the first state of player 
    state_current(1,:) = handle_state(p_cards, card_showing);
    
    s_idx = 1;
    policy_idx = sub2ind( [21-12+1,13,2], state_current(s_idx,1)-12+1, state_current(s_idx,2), state_current(s_idx,3)+1 );
    % exploring starts: choose initial action randomly 
    policy(policy_idx) = unidrnd(2)-1;
    action = policy(policy_idx);
    
    while (action == 1 && p_sum <= 21)
        p_cards = [ p_cards, deck(1) ]; 
        deck = deck(2:end); 
        [p_sum, usable_ace] = calculate_sum(p_cards); 

        state_current(end+1,:) = handle_state(p_cards, card_showing);

        if (p_sum <= 21)
            s_idx = s_idx + 1;
            policy_idx = sub2ind( [21-12+1,13,2], state_current(s_idx,1)-12+1, state_current(s_idx,2), state_current(s_idx,3)+1 ); 
            action = policy(policy_idx);
        end
    end

    % dealer's policy
    while( d_sum <= 17 )
         
        hit = dealer_case(d_cards, d_sum);
        % handle "Soft17" in two case
        if (hit == 2)
            % if dealer hit in "Soft17": hit = 1 here
            % if dealer stick in "Soft17": hit = 0 here
            hit = 1;
        end
        if (hit == 0)
            break;
        end
        d_cards = [ d_cards, deck(1) ]; 
        deck = deck(2:end); 
        d_sum = calculate_sum(d_cards);
        
    end

    % calculate the reward in this episode
    reward = calculate_reward(p_sum, d_sum);

    % calculate the action-value function and policy
    for s_idx = 1: size(state_current, 1)
        if ((state_current(s_idx, 1) >= 12) && (state_current(s_idx, 1) <= 21))
            state_idx = sub2ind( [10,13,2], state_current(s_idx,1)-12+1, state_current(s_idx,2), state_current(s_idx,3)+1 );
            action_idx = policy(state_idx) + 1;
            returns_cnt(state_idx, action_idx) = returns_cnt(state_idx, action_idx) + 1;
            returns(state_idx, action_idx) = returns(state_idx, action_idx) + reward;
            % take the average of returns and calculate Q
            Q(state_idx, action_idx) = returns(state_idx, action_idx) / returns_cnt(state_idx, action_idx);
            % record the action that maximize Q
            [maxQ_val, maxQ_idx] = max(Q(state_idx, :));
            policy(state_idx) = maxQ_idx - 1;
        end
    end
end
toc;

% plot the result
mc_value_fn = max( Q, [], 2 ); % get the state-value function from maxQ
mc_value_fn = reshape( mc_value_fn, [21-12+1,13,2]); 
if( 1 ) 
  figure; mesh( 1:13, 12:21, mc_value_fn(:,:,1) ); 
  xlabel( 'dealers showing card' ); ylabel( 'card sum' ); axis xy; %view([67,5]);
  title( 'no usable ace' ); drawnow; 
  fn=sprintf('state_value_fn_nua_%d_mesh.eps',episodes); saveas( gcf, fn, 'eps2' ); 
  figure; mesh( 1:13, 12:21,  mc_value_fn(:,:,2) ); 
  xlabel( 'dealers showing card' ); ylabel( 'card sum' ); axis xy; %view([67,5]);
  title( 'usable ace' ); drawnow; 
  fn=sprintf('state_value_fn_ua_%d_mesh.eps',episodes); saveas( gcf, fn, 'eps2' ); 
  
end

policy = reshape( policy, [21-12+1,13,2] ); 
if( 1 ) 
  figure; imagesc( 1:13, 12:21, policy(:,:,1) );  
  xlabel( 'dealers showing card' ); ylabel( 'card sum' ); axis xy; %view([67,5]);
  title( 'no usable ace' ); drawnow; 
  fn=sprintf('bj_opt_pol_nua_%d_image.eps',episodes); saveas( gcf, fn, 'eps2' ); 
  figure; imagesc( 1:13, 12:21,  policy(:,:,2) ); 
  xlabel( 'dealers showing card' ); ylabel( 'card sum' ); axis xy; %view([67,5]);
  title( 'usable ace' ); drawnow; 
  fn=sprintf('bj_opt_pol_ua_%d_mesh.eps',episodes); saveas( gcf, fn, 'eps2' ); 
end
return;


function [cs, usableAce] = calculate_sum(cards)
    % get the index of each card 
    cards_val = mod(cards - 1, 13) + 1; 
    % J, Q, K will be counts as 10
    cards_val = min( cards_val, 10 );
    current_sum = sum(cards_val); 
    % If the player holds an ace that can be count as 11 without goes bust
    % then usableAce = 1, otherwise usableAce = 0
    if (any(cards_val == 1)) && (current_sum <= 11)
       current_sum = current_sum + 10;
       usableAce = 1; 
    else
       usableAce = 0; 
    end
    cs = current_sum;
end

function [h] = dealer_case(d_cards, d_sum)
    % hit = 2 means Soft17, hit = 1 means hit, hit = 0 means stick
    if (d_sum < 17)
        h = 1;
    elseif (d_sum > 17)
        h = 0;
    elseif (any(d_cards == 1))
        h = 2;
    else
        % d_sum equal to 17 but have no ace, stick
        h = 0;
    end
end

function [reward] = calculate_reward(p_sum, d_sum)
    % rewards of +1, -1, 0 are given for winning, losing, and drawing 
    % if player's card sum equal to dealer's, it's a draw
    
    % Otherwise, if player's card sum is over 21, goes bust, player lose
    if(p_sum > 21) 
      reward = -1; 
      return; 
    end
    % if dealer's card sum is over 21, goes bust, player win
    if(d_sum > 21) 
      reward = +1; 
      return; 
    end
    if(p_sum == d_sum) 
      reward = 0; 
      return;
    end

    % the one who have a larger card sum win the game
    if(p_sum > d_sum) 
      reward = +1; 
    else
      reward = -1; 
    end
end

function [state] = handle_state(p_cards, card_showing)
    [p_sum, usable_ace] = calculate_sum(p_cards);
    card_showing = mod(card_showing - 1, 13) + 1;
    state = [p_sum, card_showing, usable_ace];
end
