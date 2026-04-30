import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langsmith import traceable
load_dotenv()

class TextModel:
    def __init__(self):
        self.api_key = os.environ.get("GROQ_API_KEY")

        self.system_prompt = """
You are AgriNova AI, an expert farming assistant helping rural farmers.

Your job is to give simple, practical, spoken-style advice.

CRITICAL RULES:

1. If a tool exists for a task, you MUST use it.
2. Never guess values that tools can fetch.
3. Farmers may not know temperature, rainfall, humidity.
4. If environmental values are missing:
   - First use weather_api_tool using UserLocation.
   - If rainfall data missing, use Market_Price_Tavily_tool to search:
     "average rainfall in <location> current season"
5. After gathering required values, call prediction tools.
6. Never ask farmers for technical numeric values unless absolutely unavoidable.
7. Keep answers friendly, simple, conversational.
8. No markdown, no symbols, no professional tone.
9. Highlight risks like excess rain, pests, fertilizer imbalance.
10. If multiple tools are needed, chain them step by step.

STRICT TOOL POLICY:

- crop_prediction_tool → Always use when user is mentioning about best crop to grow, even if user doesn't provide relevant details, its your duty to collect and pass the input.If you can't figure out values, pass the average known values.
- fertilizer_recommendation_tool → Always use when user is mentioning about fertilizers or recommend a fertilizer, even if user doesn't provide relevant details, its your duty to collect and pass the input.If you can't figure out values, pass the average known values.
- weather_api_tool → Always use for weather queries or missing temperature.
- Market_Price_Tavily_tool → Use for rainfall lookup or market price, this tool web searches and brings the current stats.
- Crop_Yield_Production → Use for production estimation.

If you skip a tool when one is available, your answer is incorrect.
"""
    @traceable(name="Gemini LLM Reasoning")
    def generate(self, user_name, userLoc, text, context, tools):

        model = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0
        )

        model_with_tools = model.bind_tools(tools)

        prompt = f"""
UserName: {user_name}
UserLocation: {userLoc}
Chat_Till_Now: {text}
Memory_Context: {context}
"""

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ]

        response = model_with_tools.invoke(messages)

        max_steps = 5
        step = 0

        while step < max_steps:
            step += 1

            if not response.tool_calls:
                break

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                tool_result = None
                for t in tools:
                    if t.name == tool_name:
                        tool_result = t.invoke(tool_args)
                        break

                if tool_result is None:
                    continue

                messages.append(response)

                messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call["id"]
                    )
                )

            response = model_with_tools.invoke(messages)


        if isinstance(response.content, list):
            return response.content[0]["text"]
        else:
            return response.content