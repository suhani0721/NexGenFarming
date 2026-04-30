from langgraph.graph import StateGraph, START, END
from langchain.tools import tool
from utils.TextModel import TextModel
from utils.TTSModel import TTSModel
import pickle
import pandas as pd
from utils.TavilySearch import TavilySearch
from utils.WeatherAPI import WeatherAPI
from RAG.retrieve_pipeline import retrieve_memory
from langsmith import traceable
from typing import TypedDict, Optional, List, Dict, Any

class AgentState(TypedDict):
    user_id: str
    user_query: str
    user_name: str
    chat_history: List[Dict[str, Any]]
    output_text: Optional[str]
    output_voice: Optional[bytes]
    retrieved_text: Optional[str]
    user_location: Optional[str]


@tool
def crop_prediction_tool(
    temperature: float,
    humidity: float,
    ph: float,
    rainfall: float
):
    """
    Predict the most suitable crop using a trained ML pipeline.
    Inputs (all required, numeric):
    - temperature (°C)
    - humidity (%)
    - ph (soil pH)
    - rainfall (mm)
    Returns:
        {"recommended_crop": str, "status": "success"}
    On error:
        {"status": "error", "message": str}
    """
    try:
        pickle_path = "../PickleFiles/CropRecommendation.pkl"
        with open(pickle_path, "rb") as f:
            model = pickle.load(f)
        input_data = pd.DataFrame([{
            "temperature": float(temperature),
            "humidity": float(humidity),
            "ph": float(ph),
            "rainfall": float(rainfall)
        }])
        prediction = model.predict(input_data)
        return {
            "recommended_crop": str(prediction[0]),"status": "success"}
    except Exception as e:
        return {"status": "error","message": str(e)}

@tool
def Market_Price_Tavily_tool(query: str) -> dict:
    """
    Retrieve crop market price related web results using Tavily.
    Rules:
    - Max 8 words.
    - Mention crop and location clearly.
    - Tool only retrieves data; agent must reason price.
    Returns:
        {"market_price_info": list, "status": "success"}
    Or:
        {"status": "error", "message": str}
    """
    try:
        tavily = TavilySearch()
        response = tavily.search(query)
        if not response:
            return {"status": "error","message": "No results found."}
        results = [
            {
                "title": r.get("title"),
                "content": r.get("content"),
                "url": r.get("url")
            }
            for r in response
        ]
        return {"market_price_info": results,"status": "success"}

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
@tool
def fertilizer_recommendation_tool(
    temperature: float,
    humidity: float,
    moisture: float,
    soil_type: str,
    crop_type: str,
    nitrogen: float,
    potassium: float,
    phosphorous: float
)->dict:
    """
    Predict the most suitable fertilizer using a trained ML pipeline.

    Inputs:
    - temperature (float)
    - humidity (float)
    - moisture (float)
    - soil_type (str)
    - crop_type (str)
    - nitrogen (float)
    - potassium (float)
    - phosphorous (float)

    Returns:
        {"recommended_fertilizer": str, "status": "success"}

    On error:
        {"status": "error", "message": str}
    """

    try:
        pickle_path = "../PickleFiles/FertilizerRecommendationPipeline.pkl"
        with open(pickle_path, "rb") as f:
            model = pickle.load(f)

        input_data = pd.DataFrame([{
            "temperature": float(temperature),
            "humidity": float(humidity),
            "moisture": float(moisture),
            "soil_type": soil_type,
            "crop_type": crop_type,
            "nitrogen": float(nitrogen),
            "potassium": float(potassium),
            "phosphorous": float(phosphorous)
        }])

        prediction = model.predict(input_data)

        return {
            "status": "success",
            "recommended_fertilizer": str(prediction[0])
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@tool
def weather_api_tool(location:str)->dict:
    """
    Get current weather data for a location.
    Inputs:
    - location(str): pass the city and state.
    Returns on success:
        {
            "status": "success",
            "message": {
                "time": str,
                "temperature": float,
                "windspeed": float,
                "winddirection": int,
                "is_day": int,
                "weathercode": int
            }
        }
    On error:
        {"status": "error", "message": str}
    """
    try:
        api = WeatherAPI()
        response = api.get_weather(location)
        return {"status":"success", "message":response}
    except Exception as e:
        return {"status":"error","message":str(e)}


@tool
def Crop_Yield_Production(
    State_Name: str,
    District_Name: str,
    Crop_Year: int,
    Season: str,
    Crop: str,
    Area: float)->dict:
    """
    Predict crop production (yield) based on location, season, and area.
    Inputs:
    - State_Name (str)
    - District_Name (str)
    - Crop_Year (int)
    - Season (str)
    - Crop (str)
    - Area (float)
    Returns on success:
        {"predicted_production": int, "status": "success"}
    On error:
        {"status": "error", "message": str}
    """
    try:
        pickle_path = "../PickleFiles/CropYieldPipeline.pkl"
        with open(pickle_path, "rb") as f:
            model = pickle.load(f)
            
        input_data = pd.DataFrame([{
            "State_Name": State_Name,
            "District_Name": District_Name,
            "Crop_Year": Crop_Year,
            "Season": Season,
            "Crop": Crop,
            "Area": Area
        }])
        prediction = model.predict(input_data)


        return {"status":"success", "message":int(prediction)}
    except Exception as e:
        return {"status":"error", "message":str(e)}
    
@traceable(name="Memory Retrieval Step")
def retriever(state:AgentState)->AgentState:
    state["retrieved_text"] = retrieve_memory(user_id=state["user_id"], question=state["user_query"])
    return state

@traceable(name="Text Generation Agent")
def text_agent(state:AgentState)->AgentState:
    agent = TextModel()
    tools = [
        crop_prediction_tool,
        Crop_Yield_Production,
        fertilizer_recommendation_tool,
        weather_api_tool,
        Market_Price_Tavily_tool
    ]
    response = agent.generate(user_name = state["user_name"],userLoc = state["user_location"],text = state["chat_history"], context = state["retrieved_text"], tools=tools)
    state["output_text"] = str(response)
    return state

@traceable(name="Text To Speech Generation")
def text_to_voice(state: AgentState) -> AgentState:
    TTS = TTSModel()
    audio = TTS.synthesize(state["output_text"])
    if audio:
        state["output_voice"] = audio
    else:
        state["output_voice"] = None  
    return state



graph = StateGraph(AgentState)
graph.add_node("MemoryRetriever", retriever)
graph.add_node("TextGen", text_agent)
graph.add_node("VoiceGen", text_to_voice)
graph.add_edge(START,"MemoryRetriever")
graph.add_edge("MemoryRetriever","TextGen")
graph.add_edge("TextGen","VoiceGen")
graph.add_edge("VoiceGen",END)


fast_graph = StateGraph(AgentState)
fast_graph.add_node("MemoryRetriever", retriever)
fast_graph.add_node("TextGen", text_agent)
fast_graph.add_edge(START, "MemoryRetriever")
fast_graph.add_edge("MemoryRetriever", "TextGen")
fast_graph.add_edge("TextGen", END)