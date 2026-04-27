import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()
class LLM_Reasoning:
    def __init__(self):
        self.client = InferenceClient(
            api_key=os.getenv("HUGGING_FACE_API_KEY"),
        )

    def llm_brain(self, prompt):
        completion = self.client.chat.completions.create(
        model=os.getenv("META_LLAMA_MODEL"),
        messages=[
            {
            "role": "user",
            "content": content(str)
            }
        ],
    )
    print(completion.choices[0].message)