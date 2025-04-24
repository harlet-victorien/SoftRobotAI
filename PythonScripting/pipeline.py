from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
import aiModules as ai

app = Flask(__name__)
session = ort.InferenceSession("Models/RobotAgent5.onnx")

# Endpoint to return a vector
@app.route('/get_vector', methods=['POST'])
def get_vector():
    data = request.json
    message = data.get('message', '')
    print(f"Message reçu: {message}")
    real_input = message.split(", ")
    x = float(real_input[0].replace(",", "."))
    y = float(real_input[1].replace(",", "."))
    z = float(real_input[2].replace(",", "."))

    real_input = [np.array([x, y, z], dtype=np.float32)]
    print(f"Message split: {real_input}")
    # Faire l'inférence
    outputs = session.run(None, {session.get_inputs()[0].name: real_input})
    outputs = outputs[2][0].tolist()
    for i in range(len(outputs)):
        outputs[i] = (outputs[i] + 1) * 7.5
    print(f"Message envoyé: {outputs}")
    return jsonify({"vector": outputs})

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

