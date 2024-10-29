import json
from openai import OpenAI

def load_json(filepath: str) -> dict:
    """Load and return JSON data from a given file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def initialize_client() -> OpenAI:
    """Initialize and return the OpenAI client with the required API credentials."""
    keys = load_json("configs/keys.json")
    return OpenAI(
        api_key=keys["api_key"],
        organization=keys["organization"],
        project=keys["project"]
    )

client = initialize_client()
settings = load_json("configs/settings.json")

def generate_completion(user_text: str, system_text: str) -> str:
    """
    Generate a completion using OpenAI's chat completion API.
    
    Args:
        user_text (str): The user-provided input text.
        system_text (str): The system's initial guidance text.
    
    Returns:
        str: The content of the AI's response.
    """
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text}
        ],
        temperature=0.0
    )
    return completion.choices[0].message.content

def get_task_steps(user_prompt: str) -> str:
    """
    Get a list of steps required to complete a task based on the user prompt.
    
    Args:
        user_prompt (str): The prompt describing the task.

    Returns:
        str: Steps to complete the task.
    """
    steps_prompt = f"""\
List the steps required to complete the following task. The steps will be excecuted in code on the user's computer. Do not write any code. You can use the code to detect things if it wasn't specified in the prompt or something is unclear. There can be as many steps as needed. Be as detailed as possible to minimize failure. Complete the tasks as best as you can. If the task is unclear, print out a response and then exit.
USER: {user_prompt}"""
    
    system_text = "You are an assistant that breaks down tasks into clear, sequential steps."
    steps_response = generate_completion(steps_prompt, system_text)
    
    return steps_response

def get_python_script_from_prompt(user_prompt: str, steps: str) -> str:
    """
    Generate a Python script based on the user prompt and the steps required to complete the task.

    Args:
        user_prompt (str): The prompt describing the desired Python script's functionality.
        steps (str): The steps needed to complete the task as outlined by the AI.

    Returns:
        str: A Python script generated from the AI's response.
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
    """
    Confirm with the user before executing the script and execute if confirmed.

    Args:
        script (str): The Python script to execute.
    """
    if settings.get("confirm_before_run"):
        confirmation = input(f"{script}\n\n> Press Enter to confirm run. Type 'NO' and then press Enter to abort: ")
        if confirmation.strip().upper() == "NO":
            print("Aborted")
            return

    exec(script)

if __name__ == "__main__":
    user_request = input("What would you like the AI to do?\n")
    steps = get_task_steps(user_request)
    # print("Steps to complete the task:\n", steps)
    script = get_python_script_from_prompt(user_request, steps)
    confirm_and_execute_script(script)
