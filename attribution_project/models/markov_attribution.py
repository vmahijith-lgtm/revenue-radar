import pandas as pd
from collections import defaultdict
from itertools import chain

def model(dbt, session):
    dbt.config(materialized='table')
    
    df = dbt.source("main", "raw_clicks").df()
    
    if df.empty:
        return pd.DataFrame({
            'channel': ['ERROR'],
            'attributed_value': [0.0],
            'status': ['failed'],
            'error_msg': ['Empty dataframe']
        })
    
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
    
    # Build conversion journeys
    journeys = []
    for user_id, group in df.groupby('user_id'):
        path = group['channel'].tolist()
        converted = group['conversion'].max()
        conv_value = group[group['conversion'] == 1]['conversion_value'].sum()
        
        if converted == 1 and len(path) > 0:
            journeys.append({
                'path': tuple(path), 
                'conversion_value': conv_value,
                'count': 1
            })
    
    if len(journeys) == 0:
        return pd.DataFrame({
            'channel': ['ERROR'],
            'attributed_value': [0.0],
            'status': ['failed'],
            'error_msg': ['No conversions']
        })
    
    # Get all unique channels
    channels = list(set(chain.from_iterable([list(j['path']) for j in journeys])))
    
    # Build transition counts: START -> channel1 -> channel2 -> CONVERSION
    transitions = defaultdict(lambda: defaultdict(int))
    
    for journey in journeys:
        path = ['START'] + list(journey['path']) + ['CONVERSION']
        for i in range(len(path) - 1):
            transitions[path[i]][path[i+1]] += 1
    
    # Convert to transition probabilities
    trans_prob = {}
    for from_state, to_dict in transitions.items():
        total = sum(to_dict.values())
        trans_prob[from_state] = {
            to_state: count / total 
            for to_state, count in to_dict.items()
        }
    
    # Calculate removal effect for each channel
    def calculate_conversion_probability(transition_probs, excluded_channel=None):
        """
        Calculate probability of reaching CONVERSION from START.
        Uses simple path counting approach.
        """
        # Build filtered transition probabilities
        filtered_trans = {}
        
        for from_state, to_dict in transition_probs.items():
            # Skip if this state is the excluded channel
            if from_state == excluded_channel:
                continue
            
            # Filter out transitions TO the excluded channel
            filtered_to = {
                to_state: prob 
                for to_state, prob in to_dict.items() 
                if to_state != excluded_channel
            }
            
            # Renormalize probabilities
            total_prob = sum(filtered_to.values())
            if total_prob > 0:
                filtered_trans[from_state] = {
                    to_state: prob / total_prob 
                    for to_state, prob in filtered_to.items()
                }
        
        # Simple approach: count how many conversion paths remain
        # More sophisticated: use matrix algebra or Monte Carlo simulation
        # For now, we'll use a simplified removal effect
        
        if excluded_channel is None:
            # Base case: count all conversions
            return len(journeys)
        else:
            # Count conversions that don't involve the excluded channel
            remaining = sum(1 for j in journeys if excluded_channel not in j['path'])
            return remaining
    
    # Calculate base conversion count
    base_conversions = calculate_conversion_probability(trans_prob, excluded_channel=None)
    
    # Calculate removal effect for each channel
    removal_effects = {}
    for channel in channels:
        conversions_without = calculate_conversion_probability(trans_prob, excluded_channel=channel)
        # Removal effect = conversions lost when channel is removed
        removal_effects[channel] = base_conversions - conversions_without
    
    # Handle negative removal effects (shouldn't happen, but defensive)
    removal_effects = {ch: max(0, effect) for ch, effect in removal_effects.items()}
    
    # Normalize to get attribution weights
    total_removal_effect = sum(removal_effects.values())
    
    if total_removal_effect == 0:
        # Fallback: equal distribution
        attribution_weights = {ch: 1.0 / len(channels) for ch in channels}
    else:
        attribution_weights = {
            ch: removal_effect / total_removal_effect 
            for ch, removal_effect in removal_effects.items()
        }
    
    # Calculate total conversion value to distribute
    total_value = sum([j['conversion_value'] for j in journeys])
    
    # Build result
    result_data = []
    for channel in channels:
        weight = attribution_weights[channel]
        attributed_val = round(total_value * weight, 2)
        
        result_data.append({
            'channel': channel,
            'attributed_value': attributed_val,
            'removal_effect': removal_effects[channel],
            'attribution_weight': round(weight, 4),
            'status': 'success',
            'error_msg': None
        })
    
    result_df = pd.DataFrame(result_data)
    result_df = result_df.sort_values('attributed_value', ascending=False)
    
    return result_df