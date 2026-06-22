# 🌱 NexGenFarming

An AI-powered smart farming assistant that helps farmers make informed agricultural decisions using Machine Learning, Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and real-time information retrieval.

## 🚀 Features

- 🌾 Crop Recommendation based on soil and environmental conditions
- 🌿 Fertilizer Recommendation
- 📈 Crop Yield Prediction
- 🤖 AI-powered Farming Assistant
- 🧠 RAG-based agricultural knowledge retrieval
- 🌍 Location-aware farming guidance
- 💬 Conversational chatbot for farming queries
- 🔍 Web search integration for up-to-date agricultural information
- 📚 Conversation history management

---

## 🛠️ Tech Stack

### Programming Language
- Python

### Frameworks
- Flask
- LangGraph
- LangChain

### AI & Machine Learning
- Large Language Models (LLMs)
- Retrieval-Augmented Generation (RAG)
- Sentence Transformers
- FAISS
- Scikit-learn
- XGBoost

### Database
- MongoDB

### Tools & Libraries
- Hugging Face Transformers
- Groq API
- Tavily Search API
- Git
- Jupyter Notebook

---

## 📂 Project Structure

```
NexGenFarming
│
├── app.py
├── requirements.txt
├── templates/
├── static/
├── utils/
├── RAG/
├── Datasets/
├── ModelTrainers/
├── PickleFiles/
└── README.md
```

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/your-username/NexGenFarming.git
cd NexGenFarming
```

### Create a virtual environment

```bash
python3 -m venv venv
```

### Activate the virtual environment

#### macOS/Linux

```bash
source venv/bin/activate
```

#### Windows

```bash
venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root and configure the required variables.

Example:

```env
MONGO_URI=your_mongodb_connection_string
SECRET_KEY=your_secret_key
groq_api_key=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
HF_TOKEN=your_huggingface_token   # Optional
```

---

## ▶️ Run the Application

```bash
python app.py
```

The application will be available at:

```
http://127.0.0.1:10001
```

---

## 📊 Machine Learning Models

The project includes models for:

- Crop Recommendation
- Fertilizer Recommendation
- Crop Yield Prediction

---

## 🤖 AI Capabilities

- Conversational AI Assistant
- Retrieval-Augmented Generation (RAG)
- Semantic Search using FAISS
- Agricultural Knowledge Base
- Real-time Web Search Integration

---

## 📌 Future Enhancements

- Weather Forecast Integration
- Voice-based Farmer Assistant
- Disease Detection using Computer Vision
- Market Price Prediction
- Multi-language Support
- Mobile Application

---

## 📄 License

This project is intended for educational and research purposes.
