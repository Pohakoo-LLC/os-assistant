import json
from openai import OpenAI
import google.generativeai as genai

def load_json(filepath: str) -> dict:
    """Load and return JSON data from a given file."""
    with open(filepath, 'r') as f:
        return json.load(f)

keys = load_json("configs/keys.json")
settings = load_json("configs/settings.json")

def initialize_client() -> OpenAI:
    """Initialize and return the OpenAI client with the required API credentials."""
    return OpenAI(
        api_key=keys["api_key"],
        organization=keys["organization"],
        project=keys["project"]
    )

client = initialize_client()

def generate_oai_completion(user_text: str, system_text: str) -> str:
    """Generate a completion using OpenAI's chat completion API."""
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text}
        ],
        temperature=0.0
    )
    return completion.choices[0].message.content

def generate_gemini_completion(prompt, system_prompt):
    """Generate a completion using Gemini API."""
    prompt = f"{system_prompt}\n{prompt}"
    genai.configure(api_key=keys["gemini_api_key"])
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    text = ""
    for part in response.candidates[0].content.parts:
        text += part.text
    return text

def generate_completion(user_text: str, system_text: str) -> str:
    """
    Wrapper function to generate a completion based on the configured model.
    
    Args:
        user_text (str): The user-provided input text.
        system_text (str): The system's initial guidance text.
    
    Returns:
        str: The content of the AI's response.
    """
    if settings.get("use_model") == "gemini":
        return generate_gemini_completion(user_text, system_text)
    else:
        return generate_oai_completion(user_text, system_text)

def get_task_steps(user_prompt: str) -> str:
    """Get a list of steps required to complete a task based on the user prompt."""
    steps_prompt = f"""\
List the steps required to complete the following task. The steps will be executed in code on the user's computer. Do not write any code. You can use the code to detect things if it wasn't specified in the prompt or something is unclear. There can be as many steps as needed. Be as detailed as possible to minimize failure. Complete the tasks as best as you can. If the task is unclear, print out a response and then exit. Don't do anything risky or possibly destructive.
USER: {user_prompt}"""
    
    system_text = "You are an assistant that breaks down tasks into clear, sequential steps."
    return generate_completion(steps_prompt, system_text)

def get_python_script_from_prompt(user_prompt: str, steps: str) -> str:
    """
    Generate a Python script based on the user prompt and the steps required to complete the task.
    """
    prompt = f"""\
SYSTEM: The user will ask you to complete a task on their computer. Follow these steps to complete the task:
{steps}
Then, generate a Python script that completes the task. Print that the task has been completed when the task finishes. If the user's request is unclear, have the Python script print out a response and then exit. Do not assume the cwd, use absolute paths.
USER: {user_prompt}
    """
    system_text = f"You are an assistant that generates Python scripts to run within a {settings['os']} computer."
    gpt_output = generate_completion(prompt, system_text)

    # Extract and clean the Python script from the generated response.
    script = gpt_output.split("```")[1]
    if script.startswith(("python", "py")):
        script = script.split('\n', 1)[1]
    
    return script

def confirm_and_execute_script(script: str) -> None:
    """Confirm with the user before executing the script and execute if confirmed."""
    if settings.get("confirm_before_run"):
        confirmation = input(f"{script}\n\n> Press Enter to confirm run. Type 'NO' and then press Enter to abort: ")
        if confirmation.strip().upper() == "NO":
            print("Aborted")
            return

    exec(script)

def main(request: str) -> None:
    steps = get_task_steps(request)
    script = get_python_script_from_prompt(request, steps)
    confirm_and_execute_script(script)

if __name__ == "__main__":
    request = input("Enter the task you want to complete: ")
    main(request)