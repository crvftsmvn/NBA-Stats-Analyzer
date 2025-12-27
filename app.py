"""
NBA Stats Analyzer - A comprehensive Flask web application for analyzing NBA team and player statistics.
Features: Team comparison, player analysis, top performers ranking, and game-by-game statistics.
"""

from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import json
import ast
import os
from collections import Counter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False  # Maintain JSON key order for better readability

def load_and_process_data():
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'NBA.csv')
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y %H:%M')
    
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
                if isinstance(player, (list, tuple)) and len(player) > 0:
                    # Print all player names and corresponding minutes in this home_players list
                    print(f"DEBUG: home_team {row['Home']}, opponent {row['Away']}, player_list={repr(player[0])}, minutes={repr(player[1]) if len(player)>1 else 'N/A'}")
                if isinstance(player, (list, tuple)) and len(player) > 0 and player[0] == player_name:
                    print(f"MATCH FOUND for {player_name} in home {row['Home']} vs {row['Away']}: player={player}")
                    minutes_val = safe_int(player[1]) if len(player) > 1 else 0
                    player_stats.append({
                        'date': row['Date'].strftime('%Y-%m-%d'),
                        'team': row['Home'],
                        'opponent': row['Away'],
                        'minutes': minutes_val,
                        'points': safe_int(player[2]) if len(player) > 2 else 0,
                        'rebounds': safe_int(player[3]) if len(player) > 3 else 0,
                        'assists': safe_int(player[4]) if len(player) > 4 else 0
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
                if isinstance(player, (list, tuple)) and len(player) > 0:
                    print(f"DEBUG: away_team {row['Away']}, opponent {row['Home']}, player_list={repr(player[0])}, minutes={repr(player[1]) if len(player)>1 else 'N/A'}")
                if isinstance(player, (list, tuple)) and len(player) > 0 and player[0] == player_name:
                    print(f"MATCH FOUND for {player_name} in away {row['Away']} vs {row['Home']}: player={player}")
                    minutes_val = safe_int(player[1]) if len(player) > 1 else 0
                    player_stats.append({
                        'date': row['Date'].strftime('%Y-%m-%d'),
                        'team': row['Away'],
                        'opponent': row['Home'],
                        'minutes': minutes_val,
                        'points': safe_int(player[2]) if len(player) > 2 else 0,
                        'rebounds': safe_int(player[3]) if len(player) > 3 else 0,
                        'assists': safe_int(player[4]) if len(player) > 4 else 0
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
        
        # Sort by date descending (most recent first) and get the last n games
        player_stats['date'] = pd.to_datetime(player_stats['date'])
        player_stats = player_stats.sort_values('date', ascending=False).head(n_games)
        
        if len(player_stats) == 0:
            return jsonify({'error': 'No recent games found for player'}), 404
        
        # Calculate averages using safe conversion
        averages = {
            'minutes': float(round(safe_float(player_stats['minutes'].mean()), 1)),
            'points': float(round(safe_float(player_stats['points'].mean()), 1)),
            'rebounds': float(round(safe_float(player_stats['rebounds'].mean()), 1)),
            'assists': float(round(safe_float(player_stats['assists'].mean()), 1))
        }
        
        # Prepare stats for chart
        stats = []
        for _, row in player_stats.iterrows():
            try:
                # Convert date back to string format if it's a datetime object
                date_str = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])
                stats.append({
                    'date': date_str,
                    'team': row['team'],
                    'opponent': row['opponent'],
                    'minutes': float(safe_float(row['minutes'])),
                    'points': float(safe_float(row['points'])),
                    'rebounds': float(safe_float(row['rebounds'])),
                    'assists': float(safe_float(row['assists']))
                })
            except Exception as e:
                print(f"Error processing row: {row}")
                print(f"Error details: {str(e)}")
                continue
        
        # Get predictions
        predictions = {}
        for stat_type in ['minutes', 'points', 'rebounds', 'assists']:
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
                            player_stats[name] = {'points': [], 'rebounds': [], 'assists': []}
                        
                        player_stats[name]['points'].append(safe_int(player[2]) if len(player) > 2 else 0)
                        player_stats[name]['rebounds'].append(safe_int(player[3]) if len(player) > 3 else 0)
                        player_stats[name]['assists'].append(safe_int(player[4]) if len(player) > 4 else 0)
            
            except Exception as e:
                print(f"Error processing game: {str(e)}")
                print(f"Game data: {game}")
                continue
        
        # Calculate averages and sort
        rankings = {
            'points': [],
            'rebounds': [],
            'assists': []
        }
        
        for player, stats in player_stats.items():
            for stat_type in ['points', 'rebounds', 'assists']:
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

@app.route('/team_top_players', methods=['POST'])
def team_top_players():
    """Returns the top 5 players (sum of points+rebounds+assists) for the last N games of the selected team (where N = n_games from POST data)."""
    try:
        data = request.json
        team = data.get('team')
        n_games = safe_int(data.get('n_games', 5))

        if not team:
            return jsonify({'error': 'Team name is required'}), 400
        if n_games <= 0:
            n_games = 5

        df = load_and_process_data()
        # Only these N most recent games are considered below
        team_games = get_team_games(df, team, n_games)

        totals = {}
        for _, game in team_games.iterrows():
            try:
                pdata = game['HomeD'] if game['Home'] == team else game['AwayD']
                if isinstance(pdata, str):
                    try:
                        players = ast.literal_eval(pdata)
                    except (ValueError, SyntaxError):
                        continue
                else:
                    players = pdata
                for player in players:
                    if isinstance(player, (list, tuple)) and len(player) >= 5:
                        name = str(player[0])
                        pts = safe_int(player[2]) if len(player) > 2 else 0
                        reb = safe_int(player[3]) if len(player) > 3 else 0
                        asts = safe_int(player[4]) if len(player) > 4 else 0
                        if name not in totals:
                            totals[name] = {'player': name, 'points': 0, 'rebounds': 0, 'assists': 0, 'total': 0}
                        totals[name]['points'] += pts
                        totals[name]['rebounds'] += reb
                        totals[name]['assists'] += asts
                        totals[name]['total'] += pts + reb + asts
            except Exception:
                continue
        # Sorted by total over those last n_games, top-5
        ranking = sorted(totals.values(), key=lambda x: x['total'], reverse=True)[:5]
        return jsonify({'team': team, 'top_players': ranking})
    except Exception as e:
        error_msg = f"Error in team_top_players: {str(e)}"
        print(error_msg)
        if 'data' in locals():
            print(f"Request data: {data}")
        return jsonify({
            'error': error_msg,
            'request_data': data if 'data' in locals() else None
        }), 500


# --------------------- Player Profile Utilities ---------------------

def player_in_game(row, player_name):
    """Check if the player participated in the given game row."""
    try:
        for col in ['HomeD', 'AwayD']:
            pdata = row[col]
            if isinstance(pdata, str):
                try:
                    pdata = ast.literal_eval(pdata)
                except (ValueError, SyntaxError):
                    continue
            if isinstance(pdata, list):
                # player name is first element in each sublist
                for p in pdata:
                    if isinstance(p, (list, tuple)) and len(p) > 0 and p[0] == player_name:
                        return True
    except Exception:
        pass
    return False

def compute_team_result(row, team):
    """Return 'W' or 'L' for the given team in the game row and points diff."""
    try:
        if row['Home'] == team:
            team_pts, opp_pts = safe_int(row['HomeP']), safe_int(row['AwayP'])
        else:
            team_pts, opp_pts = safe_int(row['AwayP']), safe_int(row['HomeP'])
        return ('W' if team_pts > opp_pts else 'L', team_pts, opp_pts)
    except Exception:
        return ('N/A', 0, 0)

def get_team_performance(df, team, player_name, with_player=True):
    """Aggregate team performance with or without the player."""
    games = df[(df['Home'] == team) | (df['Away'] == team)]
    records = []
    for _, row in games.iterrows():
        participated = player_in_game(row, player_name)
        if participated != with_player:
            continue
        result, team_pts, opp_pts = compute_team_result(row, team)
        records.append({
            'date': row['Date'].strftime('%Y-%m-%d'),
            'opponent': row['Away'] if row['Home'] == team else row['Home'],
            'team_pts': team_pts,
            'opp_pts': opp_pts,
            'result': result
        })
    wins = sum(1 for r in records if r['result'] == 'W')
    losses = sum(1 for r in records if r['result'] == 'L')
    avg_for = round(np.mean([r['team_pts'] for r in records]), 1) if records else 0.0
    avg_against = round(np.mean([r['opp_pts'] for r in records]), 1) if records else 0.0
    return {
        'wins': wins,
        'losses': losses,
        'avg_for': avg_for,
        'avg_against': avg_against,
        'games': records
    }

def get_team_core_players(df, team, min_ratio=0.6):
    """Return list of core players for the team (appear in >= min_ratio of games)."""
    # Count appearances
    games = df[(df['Home'] == team) | (df['Away'] == team)]
    total_games = len(games)
    counts = Counter()
    for _, row in games.iterrows():
        pdata = row['HomeD'] if row['Home'] == team else row['AwayD']
        if isinstance(pdata, str):
            try:
                pdata = ast.literal_eval(pdata)
            except (ValueError, SyntaxError):
                continue
        for p in pdata:
            if isinstance(p, (list, tuple)) and p:
                counts[str(p[0])] += 1
    core = [name for name, c in counts.items() if c / total_games >= min_ratio]
    return core

def summarise_opponent_effect(player_games):
    """Return per-opponent averages for player performance."""
    summary = {}
    for _, row in player_games.iterrows():
        opp = row['opponent']
        pts = safe_int(row['points'])
        if opp not in summary:
            summary[opp] = []
        summary[opp].append(pts)
    return [{'opponent': k, 'games': len(v), 'avg_points': round(np.mean(v), 1)} for k, v in summary.items()]

@app.route('/get_lineup')
def get_lineup():
    """Return lineups for a specific game (by date + home + away team names)."""
    try:
        date_str = request.args.get('date')
        home = request.args.get('home')
        away = request.args.get('away')
        if not date_str or not home or not away:
            return jsonify({'error': 'Missing parameters'}), 400
        df = load_and_process_data()
        date_dt = pd.to_datetime(date_str)
        row = df[(df['Date'].dt.strftime('%Y-%m-%d') == date_str) & (df['Home']==home) & (df['Away']==away)]
        if row.empty:
            return jsonify({'error': 'Game not found'}), 404
        row = row.iloc[0]
        def parse_players(raw):
            if isinstance(raw, str):
                try:
                    raw = ast.literal_eval(raw)
                except Exception:
                    return []
            players = []
            for p in raw:
                if isinstance(p, (list, tuple)) and len(p)>=5:
                    players.append({
                        'name': str(p[0]),
                        'minutes': safe_int(p[1]),
                        'points': safe_int(p[2]),
                        'rebounds': safe_int(p[3]),
                        'assists': safe_int(p[4])
                    })
            return players
        return jsonify({
            'date': date_str,
            'home': home,
            'away': away,
            'home_players': parse_players(row['HomeD']),
            'away_players': parse_players(row['AwayD'])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/player_profile/<player_name>')
def player_profile(player_name):
    """Render detailed profile page for selected player, including games where they under- or over-performed"""
    df = load_and_process_data()
    player_games = get_player_stats(df, player_name)
    if player_games.empty:
        return f"No data found for {player_name}", 404

    # Determine player’s primary team (most frequent)
    primary_team = Counter(player_games['team']).most_common(1)[0][0]

    # Team performance aggregates
    perf_with = get_team_performance(df, primary_team, player_name, with_player=True)
    perf_without = get_team_performance(df, primary_team, player_name, with_player=False)

    # Baseline averages
    avg_pts = player_games['points'].astype(float).mean()
    std_pts = player_games['points'].astype(float).std(ddof=0)
    threshold = max(5.0, std_pts if not np.isnan(std_pts) else 5.0)

    # Identify team core players (appear in ≥60 % of team games) to detect absences
    core_players = get_team_core_players(df, primary_team)

    def find_game_row(game_date_str, team):
        """Locate the original df row for the given team/date."""
        mask = (df['Date'].dt.strftime('%Y-%m-%d') == game_date_str) & ((df['Home'] == team) | (df['Away'] == team))
        subset = df[mask]
        if len(subset):
            return subset.iloc[0]
        return None

    def players_in_df_row(row, team):
        pdata = row['HomeD'] if row['Home'] == team else row['AwayD']
        if isinstance(pdata, str):
            try:
                pdata = ast.literal_eval(pdata)
            except (ValueError, SyntaxError):
                pdata = []
        return [str(p[0]) for p in pdata if isinstance(p, (list, tuple)) and p]

    under_perf, over_perf = [], []
    for _, pg in player_games.iterrows():
        pts = safe_float(pg['points'])
        diff = pts - avg_pts
        if abs(diff) < threshold:
            continue  # Not a significant deviation
        game_row = find_game_row(pg['date'], pg['team'])
        if game_row is None:
            continue
        # Team points/result for that game
        result, team_pts, opp_pts = compute_team_result(game_row, pg['team'])
        present_players = players_in_df_row(game_row, pg['team'])
        missing_core = [p for p in core_players if p not in present_players]

        entry = {
            'date': pg['date'],
            'opponent': pg['opponent'],
            'points': pts,
            'diff': round(diff, 1),
            'team_pts': team_pts,
            'opp_pts': opp_pts,
            'result': result,
            'missing_core': missing_core
        }
        if diff < 0:
            under_perf.append(entry)
        else:
            over_perf.append(entry)

    # Convert to list­-of-dicts for Jinja
    player_games_list = player_games.to_dict('records')

    return render_template('player_profile.html',
                           player_name=player_name,
                           primary_team=primary_team,
                           player_games=player_games_list,
                           perf_with=perf_with,
                           perf_without=perf_without,
                           avg_pts=round(avg_pts, 1),
                           threshold=round(threshold, 1),
                           under_perf=under_perf,
                           over_perf=over_perf)

@app.route('/game_day_analyzer')
def game_day_analyzer():
    """Render the game day analyzer page"""
    try:
        return render_template('game_day_analyzer.html')
    except Exception as e:
        logger.error(f"Error rendering game_day_analyzer: {str(e)}")
        return f"Error loading page: {str(e)}", 500

@app.route('/api/game_days', methods=['GET'])
def get_game_days():
    """Get all unique game days sorted by date (most recent first)"""
    try:
        df = load_and_process_data()
        
        # Filter for completed games only (games with valid scores)
        # A game is considered completed if it has valid HomeP and AwayP values (can be 0 or positive)
        # Convert to numeric first to handle both string and numeric types
        homep_numeric = pd.to_numeric(df['HomeP'], errors='coerce')
        awayp_numeric = pd.to_numeric(df['AwayP'], errors='coerce')
        
        # A game is completed if both scores are valid numbers (including 0)
        completed_games = df[
            (homep_numeric.notna()) & 
            (awayp_numeric.notna())
        ].copy()
        
        if completed_games.empty:
            return jsonify({'error': 'No completed games found'}), 404
        
        # Group by date (just the date part, not time)
        completed_games['DateOnly'] = completed_games['Date'].dt.date
        game_days = sorted(completed_games['DateOnly'].unique(), reverse=True)
        
        # Convert to string format for JSON serialization
        game_days_str = [str(day) for day in game_days]
        
        return jsonify({
            'game_days': game_days_str,
            'total_days': len(game_days_str)
        })
    except Exception as e:
        logger.error(f"Error in get_game_days: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/teams', methods=['GET'])
def get_teams():
    """Get all unique teams"""
    try:
        df = load_and_process_data()
        teams = sorted(list(set(df['Home'].unique()) | set(df['Away'].unique())))
        return jsonify({'teams': teams})
    except Exception as e:
        logger.error(f"Error in get_teams: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/game_day_underperformers', methods=['POST'])
def get_game_day_underperformers():
    """Get players who underperformed on a specific game day"""
    try:
        data = request.get_json()
        game_day = data.get('game_day')
        team_filter = data.get('team')  # Optional team filter
        
        if not game_day:
            return jsonify({'error': 'Game day is required'}), 400
        
        df = load_and_process_data()
        df['DateOnly'] = df['Date'].dt.date
        
        # Filter games for the specific day
        target_date = pd.to_datetime(game_day).date()
        day_games = df[df['DateOnly'] == target_date]
        
        # Filter by team if provided
        if team_filter:
            day_games = day_games[(day_games['Home'] == team_filter) | (day_games['Away'] == team_filter)]
        
        if day_games.empty:
            return jsonify({'error': 'No games found for this day'}), 404
        
        # Calculate player averages across all games (before this day)
        all_previous_games = df[df['DateOnly'] < target_date]
        player_averages = {}
        player_team_map = {}
        
        # Process all previous games to calculate averages
        for _, row in all_previous_games.iterrows():
            # Process home team players
            home_data = row['HomeD']
            if isinstance(home_data, str):
                try:
                    home_players = ast.literal_eval(home_data)
                except (ValueError, SyntaxError):
                    home_players = []
            else:
                home_players = home_data
            
            for player in home_players:
                if isinstance(player, (list, tuple)) and len(player) >= 5:
                    name = str(player[0])
                    points = safe_float(player[2]) if len(player) > 2 else 0
                    rebounds = safe_float(player[3]) if len(player) > 3 else 0
                    assists = safe_float(player[4]) if len(player) > 4 else 0
                    
                    if name not in player_averages:
                        player_averages[name] = {'points': [], 'rebounds': [], 'assists': [], 'games': 0}
                        player_team_map[name] = row['Home']
                    
                    player_averages[name]['points'].append(points)
                    player_averages[name]['rebounds'].append(rebounds)
                    player_averages[name]['assists'].append(assists)
                    player_averages[name]['games'] += 1
            
            # Process away team players
            away_data = row['AwayD']
            if isinstance(away_data, str):
                try:
                    away_players = ast.literal_eval(away_data)
                except (ValueError, SyntaxError):
                    away_players = []
            else:
                away_players = away_data
            
            for player in away_players:
                if isinstance(player, (list, tuple)) and len(player) >= 5:
                    name = str(player[0])
                    points = safe_float(player[2]) if len(player) > 2 else 0
                    rebounds = safe_float(player[3]) if len(player) > 3 else 0
                    assists = safe_float(player[4]) if len(player) > 4 else 0
                    
                    if name not in player_averages:
                        player_averages[name] = {'points': [], 'rebounds': [], 'assists': [], 'games': 0}
                        player_team_map[name] = row['Away']
                    
                    player_averages[name]['points'].append(points)
                    player_averages[name]['rebounds'].append(rebounds)
                    player_averages[name]['assists'].append(assists)
                    player_averages[name]['games'] += 1
        
        # Calculate averages
        for name in player_averages:
            stats = player_averages[name]
            if stats['games'] > 0:
                player_averages[name] = {
                    'avg_points': float(round(np.mean(stats['points']), 1)),
                    'avg_rebounds': float(round(np.mean(stats['rebounds']), 1)),
                    'avg_assists': float(round(np.mean(stats['assists']), 1)),
                    'games': stats['games']
                }
        
        # Now check players in this game day
        underperformers = []
        threshold_percentage = 0.15  # 15% below average is considered noticeable
        
        for _, row in day_games.iterrows():
            # Process home team players
            if not team_filter or row['Home'] == team_filter:
                home_data = row['HomeD']
                if isinstance(home_data, str):
                    try:
                        home_players = ast.literal_eval(home_data)
                    except (ValueError, SyntaxError):
                        home_players = []
                else:
                    home_players = home_data
                
                for player in home_players:
                    if isinstance(player, (list, tuple)) and len(player) >= 5:
                        name = str(player[0])
                        points = safe_float(player[2]) if len(player) > 2 else 0
                        rebounds = safe_float(player[3]) if len(player) > 3 else 0
                        assists = safe_float(player[4]) if len(player) > 4 else 0
                        
                        if name in player_averages and player_averages[name]['games'] >= 3:  # Need at least 3 games for average
                            avg = player_averages[name]
                            is_underperforming = False
                            underperformance_reasons = []
                            
                            # Check if significantly below average in any stat
                            if points < avg['avg_points'] * (1 - threshold_percentage):
                                is_underperforming = True
                                underperformance_reasons.append(f"Points: {points:.1f} vs {avg['avg_points']:.1f} avg")
                            
                            if rebounds < avg['avg_rebounds'] * (1 - threshold_percentage):
                                is_underperforming = True
                                underperformance_reasons.append(f"Rebounds: {rebounds:.1f} vs {avg['avg_rebounds']:.1f} avg")
                            
                            if assists < avg['avg_assists'] * (1 - threshold_percentage):
                                is_underperforming = True
                                underperformance_reasons.append(f"Assists: {assists:.1f} vs {avg['avg_assists']:.1f} avg")
                            
                            if is_underperforming:
                                underperformers.append({
                                    'player': name,
                                    'team': row['Home'],
                                    'opponent': row['Away'],
                                    'points': float(points),
                                    'rebounds': float(rebounds),
                                    'assists': float(assists),
                                    'avg_points': avg['avg_points'],
                                    'avg_rebounds': avg['avg_rebounds'],
                                    'avg_assists': avg['avg_assists'],
                                    'reasons': underperformance_reasons,
                                    'games_played': avg['games']
                                })
            
            # Process away team players
            if not team_filter or row['Away'] == team_filter:
                away_data = row['AwayD']
                if isinstance(away_data, str):
                    try:
                        away_players = ast.literal_eval(away_data)
                    except (ValueError, SyntaxError):
                        away_players = []
                else:
                    away_players = away_data
                
                for player in away_players:
                    if isinstance(player, (list, tuple)) and len(player) >= 5:
                        name = str(player[0])
                        points = safe_float(player[2]) if len(player) > 2 else 0
                        rebounds = safe_float(player[3]) if len(player) > 3 else 0
                        assists = safe_float(player[4]) if len(player) > 4 else 0
                        
                        if name in player_averages and player_averages[name]['games'] >= 3:
                            avg = player_averages[name]
                            is_underperforming = False
                            underperformance_reasons = []
                            
                            if points < avg['avg_points'] * (1 - threshold_percentage):
                                is_underperforming = True
                                underperformance_reasons.append(f"Points: {points:.1f} vs {avg['avg_points']:.1f} avg")
                            
                            if rebounds < avg['avg_rebounds'] * (1 - threshold_percentage):
                                is_underperforming = True
                                underperformance_reasons.append(f"Rebounds: {rebounds:.1f} vs {avg['avg_rebounds']:.1f} avg")
                            
                            if assists < avg['avg_assists'] * (1 - threshold_percentage):
                                is_underperforming = True
                                underperformance_reasons.append(f"Assists: {assists:.1f} vs {avg['avg_assists']:.1f} avg")
                            
                            if is_underperforming:
                                underperformers.append({
                                    'player': name,
                                    'team': row['Away'],
                                    'opponent': row['Home'],
                                    'points': float(points),
                                    'rebounds': float(rebounds),
                                    'assists': float(assists),
                                    'avg_points': avg['avg_points'],
                                    'avg_rebounds': avg['avg_rebounds'],
                                    'avg_assists': avg['avg_assists'],
                                    'reasons': underperformance_reasons,
                                    'games_played': avg['games']
                                })
        
        return jsonify({
            'game_day': game_day,
            'underperformers': underperformers,
            'total_underperformers': len(underperformers)
        })
    
    except Exception as e:
        logger.error(f"Error in get_game_day_underperformers: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/player_underperformance_count', methods=['POST'])
def get_player_underperformance_count():
    """Get how many times selected players performed below their average"""
    try:
        data = request.get_json()
        player_names = data.get('players', [])
        
        if not player_names:
            return jsonify({'error': 'No players selected'}), 400
        
        df = load_and_process_data()
        df['DateOnly'] = df['Date'].dt.date
        
        results = {}
        
        for player_name in player_names:
            # Get all games for this player
            player_stats = get_player_stats(df, player_name)
            
            if player_stats.empty:
                results[player_name] = {
                    'total_games': 0,
                    'underperformance_count': 0,
                    'underperformance_percentage': 0.0,
                    'details': []
                }
                continue
            
            # Calculate overall average
            avg_points = float(player_stats['points'].mean())
            avg_rebounds = float(player_stats['rebounds'].mean())
            avg_assists = float(player_stats['assists'].mean())
            
            threshold_percentage = 0.15
            underperformance_games = []
            
            for _, row in player_stats.iterrows():
                points = safe_float(row['points'])
                rebounds = safe_float(row['rebounds'])
                assists = safe_float(row['assists'])
                
                is_underperforming = False
                reasons = []
                
                if points < avg_points * (1 - threshold_percentage):
                    is_underperforming = True
                    reasons.append(f"Points: {points:.1f} vs {avg_points:.1f} avg")
                
                if rebounds < avg_rebounds * (1 - threshold_percentage):
                    is_underperforming = True
                    reasons.append(f"Rebounds: {rebounds:.1f} vs {avg_rebounds:.1f} avg")
                
                if assists < avg_assists * (1 - threshold_percentage):
                    is_underperforming = True
                    reasons.append(f"Assists: {assists:.1f} vs {avg_assists:.1f} avg")
                
                if is_underperforming:
                    underperformance_games.append({
                        'date': row['date'],
                        'team': row['team'],
                        'opponent': row['opponent'],
                        'points': float(points),
                        'rebounds': float(rebounds),
                        'assists': float(assists),
                        'reasons': reasons
                    })
            
            total_games = len(player_stats)
            underperformance_count = len(underperformance_games)
            
            results[player_name] = {
                'total_games': total_games,
                'underperformance_count': underperformance_count,
                'underperformance_percentage': float(round((underperformance_count / total_games * 100) if total_games > 0 else 0, 1)),
                'avg_points': avg_points,
                'avg_rebounds': avg_rebounds,
                'avg_assists': avg_assists,
                'details': underperformance_games
            }
        
        return jsonify({'results': results})
    
    except Exception as e:
        logger.error(f"Error in get_player_underperformance_count: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)