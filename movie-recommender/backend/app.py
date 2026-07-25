import os
import sys
from flask import Flask, request, jsonify, render_template

# Ensure src directory is in the path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, 'src')
sys.path.append(SRC_DIR)

from recommender import get_recommendations

# Set template and static folders explicitly for Flask
app = Flask(__name__, 
            template_folder=os.path.join(BASE_DIR, 'frontend', 'templates'),
            static_folder=os.path.join(BASE_DIR, 'frontend', 'static'))

@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/recommend', methods=['GET'])
def recommend():
    movie = request.args.get('movie')
    model = request.args.get('model')
    
    if not movie or not model:
        return jsonify({"error": "Please provide both 'movie' and 'model' parameters."}), 400
        
    result = get_recommendations(movie, model)
    
    if "error" in result:
        # Return 404 for missing movie or dataset
        return jsonify({"error": result["error"]}), 404
        
    return jsonify({
        "model": result["model"],
        "recommendations": result["recommendations"]
    })

if __name__ == '__main__':
    # Start the Flask app
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=True)
