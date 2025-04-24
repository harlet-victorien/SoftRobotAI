from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
import aiModules as ai

app = Flask(__name__)



# Endpoint to return a list of methods with parameters
@app.route('/get_methods', methods=['POST'])
def get_methods():
    try:
        # Parse the incoming JSON request
        data = request.get_json()
        print(f"data : {data}")
        message = data.get('message', '')
        print(f"message : {message}")
        positions = ''

        # Here, we return a fixed list of method calls for demonstration purposes
        methods = ai.get_answer(message, positions)
        print(f"Methods: {methods}")

        # Return the methods as a plain string
        return methods
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='127.0.0.1', port=5000, debug=True)

