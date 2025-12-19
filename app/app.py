import os
import sys
from flask import Flask, render_template, request

# --- 1. Path Definitions & Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
TEMPLATE_DIR = os.path.join(PROJECT_ROOT, 'template')
STATIC_DIR = os.path.join(PROJECT_ROOT, 'static')

# --- 2. Import Business Logic (Safe Import) ---
bl = None
import_error = None
try:
    import business_logic as bl
except Exception as e:
    # Elkapunk MINDEN hibát (nem csak ImportError-t), pl. OSErrort az NLTK-tól
    print(f"CRITICAL ERROR: Could not import business_logic. {e}")
    import_error = str(e)

# --- 3. Flask App Initialization ---
app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

# --- 4. Global Artifacts & Lazy Loading ---
artifacts = None

def get_artifacts():
    global artifacts
    if artifacts is None and bl is not None:
        print("Loading artifacts for the first time...")
        try:
            # force_regenerate=False kritikus Vercelen!
            artifacts = bl.initialize_app(force_regenerate=False)
        except Exception as e:
            print(f"Error loading artifacts: {e}")
            return None
    return artifacts

# --- 5. Web Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search')
def search():
    if bl is None:
        return render_template('results.html', results={"error": f"Server Startup Error: {import_error}"})

    data = get_artifacts()
    if data is None:
        return render_template(
            'results.html', 
            results={"error": "Models could not be loaded. Please check server logs."}
        )

    artist_name = request.args.get('artist')
    song_name = request.args.get('song')

    results = None
    if artist_name and song_name:
        try:
            results = bl.get_recommendations(
                artist_name,
                song_name, 
                data["lyrics_df"], 
                data["tfidf_matrix"]
            )
        except Exception as e:
            results = {"error": f"Error processing request: {str(e)}"}
    
    return render_template(
        'results.html', 
        artist=artist_name,
        song=song_name,
        results=results
    )

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

if __name__ == '__main__':
    # Local development entry point
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--regenerate', action='store_true')
    args = parser.parse_args()

    if bl:
        print("Running locally...")
        artifacts = bl.initialize_app(force_regenerate=args.regenerate)
        app.run(debug=True, host='0.0.0.0', port=5000)