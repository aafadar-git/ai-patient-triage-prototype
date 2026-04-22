# AI-Assisted Patient Message Triage Prototype

A human-in-the-loop, agentic AI decision-support tool for classifying and triaging patient portal messages.

## Features
- **Classification Module**: Categorizes messages by `urgency`, `type`, and `route`, appending a confidence score.
- **Rules & Safety Engine**: Evaluates messages against red-flag phrases (e.g. "chest pain", "allergic reaction") to immediately force human intervention.
- **Drafting Step**: Generates draft responses *only* for high-confidence, routine cases. Escaping constraints ensures potentially hazardous cases are never auto-replied to.
- **Dashboard & Auditing**: Maintains an action review log and dashboard metrics of how many messages would be escalated automatically.

## Data & Evaluation
- **Mock Messages**: We provide `data/mock_messages.csv` containing synthetic MyChart-like messages representing a variety of patient scenarios for quick testing out-of-the-box.
- **Dynamic Training Evaluation**: An optional offline evaluation script is provided to test the inference engine on larger datasets of longer user inputs (e.g., `train_messages_v3_dynamic.csv`).
- **KPIs**: Success metrics on validation runs focus on maintaining a high **urgent recall** rate (ensuring emergencies are never suppressed) and a stable **draft acceptance rate** (saving clinician time).

## Project Structure
- `app.py`: The Main Streamlit application containing the UI logic and visualization panels.
- `logic.py`: The routing and analytical python module detailing rules-based escalation (and containing offline dataset evaluators).
- `data/mock_messages.csv`: The dataset containing pre-calculated example messages simulating real clinical inputs.

## Setup and Running
1. Recommended: create a virtual environment (`python -m venv venv` and `source venv/bin/activate`).
2. Run `pip install streamlit pandas requests`.
3. Start the application:
```bash
streamlit run app.py
```

## Deployment & Configuration
To use **Purdue GenAI Assisted** mode, you must configure your API keys. The app dynamically prioritizes your host environment variables and effortlessly falls back to Streamlit secrets.

### Local Development
Create a `.streamlit/secrets.toml` file in the root directory:
```toml
PURDUE_GENAI_API_KEY = "your-api-key-here"
PURDUE_GENAI_MODEL = "llama3.1:latest"
```

### Streamlit Community Cloud
When deploying to Streamlit Community Cloud:
1. Navigate to your app dashboard and click **New App**.
2. Connect your GitHub repository.
3. Select **Settings** > **Secrets**.
4. Paste the exact same TOML configuration format directly into the Secrets text field. 
The application will automatically detect the secure boundary and successfully bind the GenAI engine.

### Render Deployment
To deploy this application as a Render Web Service:
1. Link your GitHub repository in the Render Dashboard.
2. Set the Build Command: `pip install -r requirements.txt`
3. Set the Start Command: `streamlit run app.py --server.port $PORT`
4. Under the **Environment** tab, explicitly add exactly the same variables:
   - Key: `PURDUE_GENAI_API_KEY` | Value: `your-api-key-here`
   - Key: `PURDUE_GENAI_MODEL` | Value: `llama3.1:latest`
