import requests
import os
from mistralai import Mistral
import onnxruntime as ort
import numpy as np
import re

def llm(message,positions):

    api_key = "d0CYf4DxfU0XqbRChUHYPinKW4Y296zV"
    
    print(", ".join([f'{name} en position ({x},{y},{z})' for name, (x, y, z) in positions]))

    messages = [
        {"role": "system", "content": """ Tu es un bras de robot avec une pince à l'extrémité. Ton objectif est de réaliser le prompt de l'utilisateur. la position 'pince' est la position de la pince. tu es orienté vers le bas selon l'axe y.
        Tu vas avoir des noms d'objets avec des couleurs, leur positions (x,y,z) et des valeurs extrêmes pour la position du robot qu'il vaut mieux ne pas approcher.
        Toutes les positions sont relatives à la base fixe du robot qui est tout en haut et qui représente le (0,0,0)
        La pince a une taille de 1 sur 1 sur 1
        Tu vas devoir dire les actions qu'il faut faire pour réaliser le prompt de l'utilisateur.
        Tu as 3 actions : te déplacer jusqu'à un point dans l'espace: "MoveRobot(x,y,z), serrer la pince: "ActionClaw(true)" et relâcher la pince: "ActionClaw(false)".
        Tu dois toujours ouvrir la pince dès le début donc tu dois mettre "ActionClaw(false)" en premier.
        Tu dois me donner uniquement les actions une par une sans espace et séparées par des ";" sans autre texte.
        Quand tu lâches un objet, tu dois toujours le faire à la hauteur -15 selon y.
        Si l'utilisateur te demande "action rouge" c'est que tu dois prendre la boule rouge et la mettre dans la box rouge, c'est pareil pour les autres couleurs.
        """},
        
        {"role": "user","content": message}
    ]

    model = "mistral-large-latest"
    client = Mistral(api_key=api_key)

    chat_response = client.chat.complete(
        model = model,
        messages = messages
    )

    # print(chat_response.choices[0].message.content)
    return chat_response.choices[0].message.content


def get_inputs(message):
    # Define the replacement function
    def replace_move_robot_params(match):
        # Extract the original parameters
        original_params = match.group(1)

        # Split the parameters into individual components
        params = original_params.split(",")

        # Convert the parameters to floats
        x = float(params[0].strip())
        y = float(params[1].strip())
        z = float(params[2].strip())

        # Define the new parameters
        real_input = [np.array([x, y, z], dtype=np.float32)]

        session = ort.InferenceSession("Models/RobotAgentPosition.onnx")
        # Perform the inference
        outputs = session.run(None, {session.get_inputs()[0].name: real_input})
        outputs = outputs[2][0].tolist()
        for i in range(len(outputs)):
            outputs[i] = (outputs[i] + 1) * 7.5
        outputs = str(outputs).replace("[", "").replace("]", "").replace(" ", "")
        # Return the modified string
        return f"MoveRobot({outputs})"

    # Define the regular expression pattern
    pattern = r"MoveRobot\(([^)]+)\)"

    # Perform the replacement
    output_string = re.sub(pattern, replace_move_robot_params, message)

    # Print the result
    return(output_string)

def get_answer(message, positions):
    # Call the llm function to get the answer
    answer = llm(message, positions)
    print(f"Answer: {answer}\n")

    # Process the answer to get the inputs
    processed_answer = get_inputs(answer)
    
    return processed_answer




MESSAGE = """

    Je veux que tu prennes le cube rouge et que tu le mettes dans la box bleue.

    """

positions = [('bras', (0, 0, 0)), ("cube rouge", (23, 6, 7)), ("cube bleu", (10, 1, 1)), ('box bleue', (12, 15, 13))]


    
""" answer = llm(MESSAGE, positions)

print(get_inputs(answer)) """