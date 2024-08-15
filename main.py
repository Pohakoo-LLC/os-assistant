import json
from openai import OpenAI

with open("configs/keys.json", 'r') as f:
    keys = json.load(f)
with open("configs/settings.json", 'r') as f:
    settings = json.load(f)

oai_key = keys["api_key"]
oai_org = keys["organization"]
oai_project = keys["project"]

client = OpenAI(
    api_key=oai_key,
    organization=oai_org,
    project=oai_project,
)

def generate_completion(user_text:str, system_text:str) -> str:
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text}
        ],
        temperature=0.0
    )

    return completion.choices[0].message.content

def generate_completion_test(a, b):
    return """
```py
    print("hello, world!")
```
"""

def get_script(user_prompt: str):
    prompt = f"""
SYSTEM: The user will ask you to complete a task on their computer. Generate a Python script that completes this task. The user will not see any text outside of the Python script. Print that the task has been completed when the task finshes. If the user's request is unclear, have the Python script print out a response and then exit.
USER: {user_prompt}
    """
    gpt_output = generate_completion(prompt, f"You are an assistant that generates Python scripts to run within a {settings['os']} computer.")

    # Get a string that is the Python script.
    script = gpt_output.split("```")[1]
    if (script.startswith("python")):
        script = script[6:]
    elif (script.startswith("py")):
        script = script[2:]

    return script

if __name__ == "__main__":
    script = get_script(input("What would you like the AI to do?\n"))
    if settings["confirm_before_run"]:
        if not input(f"{script}\n\n> Press enter to confirm run. Type NO and then press enter to abort.") == "":
            print("Aborted")
            exit()
    exec(script)