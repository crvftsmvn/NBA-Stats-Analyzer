from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import json
import ast

app = Flask(__name__)

def load_and_process_data():
    df = pd.read_csv('NBA.csv')
    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y %H:%M')
    
    # Replace NaN values with 0 for numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(0)
    
    # Replace NaN values with empty strings for object columns
    object_columns = df.select_dtypes(include=['object']).columns
    df[object_columns] = df[object_columns].fillna('')
    
    # Convert numeric columns to float and round them
    for col in numeric_columns:
        df[col] = df[col].astype(float).round(1)
    
    return df

def get_team_games(df, team, n=10):
    # Get games where team is either home or away
    team_games = df[(df['Home'] == team) | (df['Away'] == team)].sort_values('Date', ascending=False).head(n)
    
    # Replace NaN values with 0 before converting to records
    team_games = team_games.fillna(0)
    
    # Convert any numeric columns to float and round them
    numeric_columns = team_games.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        team_games[col] = team_games[col].astype(float).round(1)
    
    return team_games

def get_player_stats(df, player_name):
    player_stats = []
    for _, row in df.iterrows():
        try:
            # Process home team players
            home_data = row['HomeD']
            if isinstance(home_data, str):
                try:
                    home_players = ast.literal_eval(home_data)
                except (ValueError, SyntaxError) as e:
                    print(f"Error parsing home data: {str(e)}")
                    print(f"Raw home data: {home_data[:100]}...")
                    continue
            else:
                home_players = home_data
            
            for player in home_players:
                if isinstance(player, (list, tuple)) and len(player) > 0 and player[0] == player_name:
                    player_stats.append({
                        'date': row['Date'].strftime('%Y-%m-%d'),
                        'team': row['Home'],
                        'opponent': row['Away'],
                        'points': safe_int(player[1]),
                        'rebounds': safe_int(player[2]),
                        'assists': safe_int(player[3]),
                        'threes': safe_int(player[4])
                    })
            
            # Process away team players
            away_data = row['AwayD']
            if isinstance(away_data, str):
                try:
                    away_players = ast.literal_eval(away_data)
                except (ValueError, SyntaxError) as e:
                    print(f"Error parsing away data: {str(e)}")
                    print(f"Raw away data: {away_data[:100]}...")
                    continue
            else:
                away_players = away_data
            
            for player in away_players:
                if isinstance(player, (list, tuple)) and len(player) > 0 and player[0] == player_name:
                    player_stats.append({
                        'date': row['Date'].strftime('%Y-%m-%d'),
                        'team': row['Away'],
                        'opponent': row['Home'],
                        'points': safe_int(player[1]),
                        'rebounds': safe_int(player[2]),
                        'assists': safe_int(player[3]),
                        'threes': safe_int(player[4])
                    })
        
        except Exception as e:
            print(f"Error processing row: {str(e)}")
            print(f"Row data: {row}")
            continue
    
    return pd.DataFrame(player_stats)

def get_team_players(df, team):
    players = set()
    for _, row in df.iterrows():
        try:
            if row['Home'] == team:
                # Handle potential string or list format
                home_data = row['HomeD']
                print(f"Processing home data for {team}:")
                print(f"Data type: {type(home_data)}")
                print(f"Data preview: {str(home_data)[:200]}...")
                
                if isinstance(home_data, str):
                    try:
                        home_players = ast.literal_eval(home_data)
                        print(f"Successfully parsed home players: {len(home_players)} players found")
                    except (ValueError, SyntaxError) as e:
                        print(f"Error parsing home data: {str(e)}")
                        print(f"Raw home data: {home_data[:200]}...")
                        continue
                else:
                    home_players = home_data
                
                # Extract player names safely
                for player in home_players:
                    if isinstance(player, (list, tuple)) and len(player) > 0:
                        players.add(str(player[0]))
            
            elif row['Away'] == team:
                # Handle potential string or list format
                away_data = row['AwayD']
                print(f"Processing away data for {team}:")
                print(f"Data type: {type(away_data)}")
                print(f"Data preview: {str(away_data)[:200]}...")
                
                if isinstance(away_data, str):
                    try:
                        away_players = ast.literal_eval(away_data)
                        print(f"Successfully parsed away players: {len(away_players)} players found")
                    except (ValueError, SyntaxError) as e:
                        print(f"Error parsing away data: {str(e)}")
                        print(f"Raw away data: {away_data[:200]}...")
                        continue
                else:
                    away_players = away_data
                
                # Extract player names safely
                for player in away_players:
                    if isinstance(player, (list, tuple)) and len(player) > 0:
                        players.add(str(player[0]))
        
        except Exception as e:
            print(f"Error processing row: {str(e)}")
            print(f"Row data: {row}")
            continue
    
    return sorted(list(players))

def predict_player_stats(player_stats, stat_type):
    try:
        # Convert stats to float, handling any '-' values
        stats = [safe_float(x) for x in player_stats[stat_type].tolist()]
        stats = [x for x in stats if not np.isnan(x)]  # Remove any NaN values
        
        if len(stats) < 3:
            mean_value = np.mean(stats) if stats else 0.0
            return {
                'prediction': float(mean_value),
                'lower_bound': float(max(0, mean_value - 1)),
                'upper_bound': float(mean_value + 1),
                'limited_data': True
            }
        
        # Calculate rolling average
        rolling_avg = pd.Series(stats).rolling(window=3, min_periods=1).mean().iloc[-1]
        if np.isnan(rolling_avg):
            rolling_avg = 0.0
        
        # Calculate standard deviation
        std = np.std(stats) if len(stats) > 1 else 1.0
        if np.isnan(std):
            std = 1.0
        
        # Calculate prediction with confidence interval
        prediction = float(rolling_avg)
        lower_bound = float(max(0, prediction - 1.5 * std))
        upper_bound = float(prediction + 1.5 * std)
        
        return {
            'prediction': prediction,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'limited_data': len(stats) < 5
        }
    except Exception as e:
        print(f"Error in predict_player_stats: {str(e)}")
        print(f"Player stats for {stat_type}: {player_stats[stat_type].tolist()}")
        return {
            'prediction': 0.0,
            'lower_bound': 0.0,
            'upper_bound': 0.0,
            'limited_data': True
        }

def safe_float(value):
    """Convert a value to float, handling '-' and invalid values as 0."""
    if value == '-' or value == '' or value is None:
        return 0.0
    try:
        result = float(value)
        return 0.0 if np.isnan(result) else result
    except (ValueError, TypeError):
        return 0.0

def safe_int(value):
    """Convert a value to int, handling various input types."""
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if np.isnan(value):
            return 0
        return int(value)
    if value == '-' or value == '' or value is None:
        return 0
    try:
        # First convert to float to handle decimal strings
        result = float(str(value))
        return 0 if np.isnan(result) else int(result)
    except (ValueError, TypeError):
        return 0

@app.route('/')
def index():
    df = load_and_process_data()
    teams = sorted(list(set(df['Home'].unique()) | set(df['Away'].unique())))
    return render_template('index.html', teams=teams)

@app.route('/get_team_players/<team>')
def get_players(team):
    try:
        print(f"Getting players for team: {team}")
        df = load_and_process_data()
        print(f"Loaded data with {len(df)} rows")
        print(f"Columns in DataFrame: {df.columns.tolist()}")
        print(f"Sample of HomeD column: {df['HomeD'].iloc[0][:200]}...")
        print(f"Sample of AwayD column: {df['AwayD'].iloc[0][:200]}...")
        
        players = get_team_players(df, team)
        print(f"Found {len(players)} players for {team}")
        return jsonify(players)
    except Exception as e:
        error_msg = f"Error in get_team_players: {str(e)}"
        print(error_msg)
        return jsonify({
            'error': error_msg,
            'team': team
        }), 500

@app.route('/compare_teams', methods=['POST'])
def compare_teams():
    try:
        data = request.json
        print(f"Received compare_teams request data: {data}")
        
        team1 = data.get('team1')
        team2 = data.get('team2')
        n_games = safe_int(data.get('n_games', 5))  # Default to 5 games
        
        if not team1 or not team2:
            print("Missing team data in request")
            return jsonify({'error': 'Both teams are required'}), 400
            
        print(f"Processing teams: {team1} vs {team2}")
        print(f"n_games value: {n_games} (type: {type(n_games)})")
        
        if n_games <= 0:
            print(f"Invalid n_games value: {n_games}, defaulting to 5")
            n_games = 5
        
        # Load and process data
        try:
            df = load_and_process_data()
            print(f"Loaded data with {len(df)} rows")
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return jsonify({'error': 'Failed to load game data'}), 500
        
        # Get team games
        try:
            team1_games = get_team_games(df, team1, n_games)
            team2_games = get_team_games(df, team2, n_games)
            print(f"Found {len(team1_games)} games for {team1}")
            print(f"Found {len(team2_games)} games for {team2}")
        except Exception as e:
            print(f"Error getting team games: {str(e)}")
            return jsonify({'error': 'Failed to get team games'}), 500
        
        # Format dates as strings
        try:
            team1_games['Date'] = team1_games['Date'].dt.strftime('%Y-%m-%d')
            team2_games['Date'] = team2_games['Date'].dt.strftime('%Y-%m-%d')
        except Exception as e:
            print(f"Error formatting dates: {str(e)}")
            return jsonify({'error': 'Failed to format game dates'}), 500
        
        # Convert to records and handle any potential serialization issues
        try:
            # Convert DataFrame to dict records and ensure all numeric values are Python floats
            team1_records = json.loads(json.dumps(team1_games.to_dict('records')))
            team2_records = json.loads(json.dumps(team2_games.to_dict('records')))
            print(f"Successfully converted {len(team1_records)} records for {team1}")
            print(f"Successfully converted {len(team2_records)} records for {team2}")
        except Exception as e:
            print(f"Error converting to records: {str(e)}")
            return jsonify({'error': 'Failed to process game records'}), 500
        
        return jsonify({
            'team1_games': team1_records,
            'team2_games': team2_records
        })
        
    except Exception as e:
        error_msg = f"Error in compare_teams: {str(e)}"
        print(error_msg)
        if 'data' in locals():
            print(f"Request data: {data}")
        return jsonify({
            'error': error_msg,
            'request_data': data if 'data' in locals() else None
        }), 500

@app.route('/player_analysis', methods=['POST'])
def player_analysis():
    try:
        data = request.get_json()
        player_name = data.get('player_name')
        
        # Validate n_games parameter with detailed error reporting
        n_games = safe_int(data.get('n_games', 5))  # Default to 5 games
        print(f"Received n_games value: {n_games} (type: {type(n_games)})")
        
        if n_games <= 0:
            print(f"Invalid or zero n_games value: {n_games}, defaulting to 5")
            n_games = 5
        
        if not player_name:
            return jsonify({'error': 'Player name is required'}), 400
        
        # Load and process data
        df = load_and_process_data()
        
        # Get player's stats
        player_stats = get_player_stats(df, player_name)
        
        if player_stats.empty:
            return jsonify({'error': 'No data found for player'}), 404
        
        # Get the last n games
        player_stats = player_stats.head(n_games)
        
        if len(player_stats) == 0:
            return jsonify({'error': 'No recent games found for player'}), 404
        
        # Calculate averages using safe conversion
        averages = {
            'points': float(round(safe_float(player_stats['points'].mean()), 1)),
            'rebounds': float(round(safe_float(player_stats['rebounds'].mean()), 1)),
            'assists': float(round(safe_float(player_stats['assists'].mean()), 1)),
            'threes': float(round(safe_float(player_stats['threes'].mean()), 1))
        }
        
        # Prepare stats for chart
        stats = []
        for _, row in player_stats.iterrows():
            try:
                stats.append({
                    'date': row['date'],
                    'team': row['team'],
                    'opponent': row['opponent'],
                    'points': float(safe_float(row['points'])),
                    'rebounds': float(safe_float(row['rebounds'])),
                    'assists': float(safe_float(row['assists'])),
                    'threes': float(safe_float(row['threes']))
                })
            except Exception as e:
                print(f"Error processing row: {row}")
                print(f"Error details: {str(e)}")
                continue
        
        # Get predictions
        predictions = {}
        for stat_type in ['points', 'rebounds', 'assists', 'threes']:
            try:
                # Convert any '-' to 0 in the stats before prediction
                player_stats[stat_type] = player_stats[stat_type].apply(safe_float)
                pred = predict_player_stats(player_stats, stat_type)
                if pred is not None:
                    predictions[stat_type] = pred
                else:
                    # Provide fallback prediction if None is returned
                    predictions[stat_type] = {
                        'prediction': float(averages[stat_type]),
                        'lower_bound': float(max(0, averages[stat_type] - 1)),
                        'upper_bound': float(averages[stat_type] + 1),
                        'limited_data': True
                    }
            except Exception as e:
                print(f"Error predicting {stat_type} for {player_name}: {str(e)}")
                print(f"Current player_stats for {stat_type}: {player_stats[stat_type].tolist()}")
                # Add a fallback prediction based on simple statistics
                predictions[stat_type] = {
                    'prediction': float(averages[stat_type]),
                    'lower_bound': float(max(0, averages[stat_type] - 1)),
                    'upper_bound': float(averages[stat_type] + 1),
                    'limited_data': True
                }
        
        return jsonify({
            'averages': averages,
            'stats': stats,
            'predictions': predictions
        })
    
    except Exception as e:
        error_msg = f"Error in player_analysis: {str(e)}"
        print(error_msg)
        if 'data' in locals():
            print(f"Request data: {data}")
        return jsonify({
            'error': error_msg,
            'request_data': data if 'data' in locals() else None
        }), 500

@app.route('/team_rankings', methods=['POST'])
def team_rankings():
    try:
        data = request.json
        team = data.get('team')
        n_games = safe_int(data.get('n_games', 5))  # Default to 5 games
        
        if not team:
            return jsonify({'error': 'Team name is required'}), 400
            
        print(f"Getting rankings for team: {team}")
        print(f"n_games value: {n_games} (type: {type(n_games)})")
        
        if n_games <= 0:
            print(f"Invalid n_games value: {n_games}, defaulting to 5")
            n_games = 5
        
        df = load_and_process_data()
        team_games = get_team_games(df, team, n_games)
        
        # Collect all player stats
        player_stats = {}
        for _, game in team_games.iterrows():
            try:
                players_to_process = []
                # Handle home team players
                if game['Home'] == team:
                    home_data = game['HomeD']
                    if isinstance(home_data, str):
                        try:
                            players_to_process = ast.literal_eval(home_data)
                        except (ValueError, SyntaxError) as e:
                            print(f"Error parsing home data: {str(e)}")
                            print(f"Raw home data: {home_data[:200]}...")
                            continue
                    else:
                        players_to_process = home_data
                # Handle away team players
                else:
                    away_data = game['AwayD']
                    if isinstance(away_data, str):
                        try:
                            players_to_process = ast.literal_eval(away_data)
                        except (ValueError, SyntaxError) as e:
                            print(f"Error parsing away data: {str(e)}")
                            print(f"Raw away data: {away_data[:200]}...")
                            continue
                    else:
                        players_to_process = away_data
                
                # Process each player's stats
                for player in players_to_process:
                    if isinstance(player, (list, tuple)) and len(player) > 0:
                        name = str(player[0])
                        if name not in player_stats:
                            player_stats[name] = {'points': [], 'rebounds': [], 'assists': [], 'threes': []}
                        
                        player_stats[name]['points'].append(safe_int(player[1]))
                        player_stats[name]['rebounds'].append(safe_int(player[2]))
                        player_stats[name]['assists'].append(safe_int(player[3]))
                        player_stats[name]['threes'].append(safe_int(player[4]))
            
            except Exception as e:
                print(f"Error processing game: {str(e)}")
                print(f"Game data: {game}")
                continue
        
        # Calculate averages and sort
        rankings = {
            'points': [],
            'rebounds': [],
            'assists': [],
            'threes': []
        }
        
        for player, stats in player_stats.items():
            for stat_type in ['points', 'rebounds', 'assists', 'threes']:
                if stats[stat_type]:  # Only calculate if we have data
                    values = [x for x in stats[stat_type] if not np.isnan(x)]  # Filter out NaN values
                    if values:  # Only calculate if we have non-NaN values
                        avg = float(round(np.mean(values), 1))
                        if not np.isnan(avg):  # Only add if average is not NaN
                            rankings[stat_type].append({
                                'player': player,
                                'average': avg
                            })
        
        # Sort each ranking
        for stat_type in rankings:
            rankings[stat_type].sort(key=lambda x: x['average'], reverse=True)
        
        return jsonify(rankings)
        
    except Exception as e:
        error_msg = f"Error in team_rankings: {str(e)}"
        print(error_msg)
        if 'data' in locals():
            print(f"Request data: {data}")
        return jsonify({
            'error': error_msg,
            'request_data': data if 'data' in locals() else None
        }), 500

if __name__ == '__main__':
    app.run(debug=True) 