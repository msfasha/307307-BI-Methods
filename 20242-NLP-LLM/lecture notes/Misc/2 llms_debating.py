import requests
import json
import time
from typing import List, Dict

class LLMDebater:
    def __init__(self, model1_name: str, model2_name: str):
        self.model1_name = model1_name
        self.model2_name = model2_name
        self.conversation_history: List[Dict] = []
        self.base_url = "http://localhost:11434/api/generate"
        
    def generate_response(self, model_name: str, prompt: str) -> str:
        """Generate a response from the specified model"""
        headers = {'Content-Type': 'application/json'}
        data = {
            'model': model_name,
            'prompt': prompt,
            'stream': False
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()['response']
        except Exception as e:
            print(f"Error generating response from {model_name}: {str(e)}")
            return ""

    def format_debate_prompt(self, previous_messages: List[Dict]) -> str:
        """Format the debate context and previous messages into a prompt"""
        prompt = """You are participating in a debate to determine which AI model should be the leader. 
Consider factors like:
- Technical capabilities
- Ethical judgment
- Decision-making ability
- Resource efficiency
- Collaboration skills

Previous messages in the debate:
"""
        for msg in previous_messages:
            prompt += f"\n{msg['model']}: {msg['message']}"
            
        prompt += "\n\nMake your case or respond to the previous message. Be concise but convincing."

        return prompt

    def conduct_debate(self, num_rounds: int = 3) -> str:
        """Conduct a debate between the two models"""
        print(f"Starting debate between {self.model1_name} and {self.model2_name}")
        
        # Initial statements
        model1_initial = self.generate_response(
            self.model1_name,
            "Make an opening statement about why you should be the leader. Consider your capabilities and approach."
        )
        self.conversation_history.append({"model": self.model1_name, "message": model1_initial})
        
        model2_initial = self.generate_response(
            self.model2_name,
            "Make an opening statement about why you should be the leader. Consider your capabilities and approach."
        )
        self.conversation_history.append({"model": self.model2_name, "message": model2_initial})
        
        # Debate rounds
        for round in range(num_rounds):
            print(f"\nRound {round + 1}")
            
            # Model 1's turn
            prompt1 = self.format_debate_prompt(self.conversation_history)
            response1 = self.generate_response(self.model1_name, prompt1)
            self.conversation_history.append({"model": self.model1_name, "message": response1})
            print(f"{self.model1_name}: {response1}")
            
            # Model 2's turn
            prompt2 = self.format_debate_prompt(self.conversation_history)
            response2 = self.generate_response(self.model2_name, prompt2)
            self.conversation_history.append({"model": self.model2_name, "message": response2})
            print(f"{self.model2_name}: {response2}")
            
            time.sleep(1)  # Prevent too many rapid requests
        
        # Final decision prompt
        final_prompt = """Based on the entire debate, make a final judgment about who should be the leader.
Consider:
1. Quality of arguments
2. Demonstration of capabilities
3. Leadership potential
4. Ethical considerations

Provide your decision and reasoning."""
        
        # Get final judgment from both models
        judgment1 = self.generate_response(self.model1_name, final_prompt)
        judgment2 = self.generate_response(self.model2_name, final_prompt)
        
        return f"""
Final Judgments:

{self.model1_name}'s verdict:
{judgment1}

{self.model2_name}'s verdict:
{judgment2}
"""
    
def main():
    # Using Ollama models: deepseek-coder and llama2
    debater = LLMDebater("qwen2.5:3b", "llama3.2")
    result = debater.conduct_debate(num_rounds=3)
    print("\nDebate Results:")
    print(result)

if __name__ == "__main__":
    main()