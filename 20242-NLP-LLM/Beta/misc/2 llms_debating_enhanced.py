# The original version was produced by Claude, this version is introduced by ChatGPT
import requests
import time
from typing import List, Dict

class LLMDebater:
    def __init__(self, model1_name: str, model2_name: str, judge_model: str):
        self.model1_name = model1_name
        self.model2_name = model2_name
        self.judge_model = judge_model
        self.conversation_history: List[Dict] = []
        self.base_url = "http://localhost:11434/api/generate"

    def generate_response(self, model_name: str, prompt: str) -> str:
        """Generate a response from the specified model sequentially"""
        headers = {'Content-Type': 'application/json'}
        data = {
            'model': model_name,
            'prompt': prompt,
            'stream': False
        }

        try:
            response = requests.post(self.base_url, headers=headers, json=data)
            response.raise_for_status()
            return response.json().get('response', '')
        except Exception as e:
            print(f"Error generating response from {model_name}: {str(e)}")
            return ""

    def format_debate_prompt(self, previous_messages: List[Dict], role: str) -> str:
        """Format the debate prompt with memory summarization and specialized role prompts"""
        
        # Summarize last 3 messages to keep context short
        summary = "\n".join([f"{msg['model']}: {msg['message']}" for msg in previous_messages[-3:]])
        
        # Role-based prompting
        role_prompt = {
            "logical": "You are an AI focused on structured reasoning, logical analysis, and evidence-based arguments.",
            "persuasive": "You are an AI that prioritizes persuasion, rhetoric, and emotional appeal.",
            "judge": "You are an impartial judge analyzing the debate to determine the winner based on reasoning and persuasiveness."
        }.get(role, "")

        return f"""
You are participating in a debate to determine the superior AI leader.
Your role: {role_prompt}

Previous messages:
{summary}

Construct your response with clarity and confidence.
"""

    def conduct_debate(self, num_rounds: int = 3):
        """Conduct a step-by-step AI debate synchronously"""
        print(f"Starting debate between {self.model1_name} and {self.model2_name}")

        # Initial Statements (Turn-Based)
        print("\nOpening Statements...")
        
        model1_initial = self.generate_response(self.model1_name, "Make an opening statement about why you should be the leader.")
        self.conversation_history.append({"model": self.model1_name, "message": model1_initial})
        print(f"{self.model1_name}: {model1_initial}")
        
        model2_initial = self.generate_response(self.model2_name, "Make an opening statement about why you should be the leader.")
        self.conversation_history.append({"model": self.model2_name, "message": model2_initial})
        print(f"{self.model2_name}: {model2_initial}")

        # Debate Rounds (Sequential Execution)
        for round_num in range(1, num_rounds + 1):
            print(f"\nRound {round_num}...")

            # Model 1's turn (logical)
            model1_prompt = self.format_debate_prompt(self.conversation_history, "logical")
            response1 = self.generate_response(self.model1_name, model1_prompt)
            self.conversation_history.append({"model": self.model1_name, "message": response1})
            print(f"{self.model1_name}: {response1}")

            # Model 2's turn (persuasive)
            model2_prompt = self.format_debate_prompt(self.conversation_history, "persuasive")
            response2 = self.generate_response(self.model2_name, model2_prompt)
            self.conversation_history.append({"model": self.model2_name, "message": response2})
            print(f"{self.model2_name}: {response2}")

            time.sleep(1)  # Avoid overwhelming API

        # Final Judgment by AI Judge
        print("\nFinal Judgment...")

        judge_prompt = self.format_debate_prompt(self.conversation_history, "judge")
        judge_decision = self.generate_response(self.judge_model, judge_prompt)

        print("\nFinal Verdict:")
        print(judge_decision)

        return judge_decision

def main():
    # Using three AI models: Two debaters and one judge
    debater = LLMDebater("qwen2.5:3b", "llama3.2", "phi4")
    result = debater.conduct_debate(num_rounds=3)
    print("\nDebate Results:")
    print(result)

if __name__ == "__main__":
    main()
