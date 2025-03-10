# AI Resume Assistant

This is an AI-powered assistant that analyzes a resume (`resume.pdf`) and provides:
- **Q&A on Resume Content** (Ask questions about experience, skills, etc.)
- **AI-Powered Insights** (Summarization, analysis)
- **Auto-Generated Resume Summary** (Concise overview in 3-4 sentences)

## Features
✅ **Resume Summary Tab** - AI-generated summary of the resume.  
✅ **Resume Q&A Tab** - Ask questions about the resume.  
✅ **AI Analysis Tab** - Get insights using an AI-powered agent.  
✅ **Download Resume** - Save the analyzed resume.  

## File Structure
```
AI-Resume-Assistant
 ├── app.py          # Main Streamlit app
 ├── resume.pdf      # Resume to analyze
 ├── README.md       # Documentation
 ├── .env            # OpenAI API Key
 ├── requirements.txt # Dependencies
```

## Installation & Setup

### Clone the repository
```bash
git clone https://github.com/yourusername/AI-Resume-Assistant.git
cd AI-Resume-Assistant
```

### Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Set up OpenAI API Key
Create a `.env` file and add:
```
OPENAI_API_KEY="your-api-key-here"
```

### Run the application
```bash
streamlit run app.py
```

---

## 📖 Usage
1. **Ensure `resume.pdf` is in the project folder**.
2. **Run the app & explore tabs**:
   - **"Resume Summary"**: Generate an AI-powered summary.
   - **"Resume Q&A"**: Ask questions about the resume.
   - **"AI Analysis"**: Get deeper insights.
3. **Download the resume from the sidebar**.

## Developers

- [Aarushi Thejaswi](https://github.com/athejaswi)  
- [Anjith Prakash](https://github.com/Anuttan)  

Feel free to fork and enhance this project!
