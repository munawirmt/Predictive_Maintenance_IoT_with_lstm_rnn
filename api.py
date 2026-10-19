from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np
from model import IoTPredictiveMaintenanceLSTM

app = FastAPI(title="IoT Predictive Maintenance API")

# മോഡൽ ലോഡ് ചെയ്യുന്നു
model = IoTPredictiveMaintenanceLSTM(input_size=21, hidden_size=32, num_layers=2)
model.load_state_dict(torch.load('lstm_model.pth'))
model.eval()

class SensorInput(BaseModel):
    data: list 

@app.post("/predict")
def predict_rul(input_data: SensorInput):
    seq = np.array(input_data.data)
    seq_t = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        prediction = model(seq_t).item()
    
    prediction = max(0, int(prediction))
    
    # 💡 കമ്പനികൾ ഉപയോഗിക്കുന്ന യഥാർത്ഥ ഇൻഡസ്ട്രി സ്റ്റാൻഡേർഡ് റൂൾ:
    # സാധാരണയായി 30 സൈക്കിളിൽ താഴെ മാത്രമാണ് എഞ്ചിന് ആകെ ആയുസ്സ് (RUL) എങ്കിൽ അത് അതീവ ഗുരുതരമാണ്.
    if prediction > 75:
        health_status = "Good"
    elif prediction > 30:
        health_status = "Warning"
    else:
        health_status = "Critical"
        
    return {"remaining_useful_life_cycles": prediction, "health_status": health_status}
