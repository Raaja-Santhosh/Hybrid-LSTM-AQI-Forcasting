import sys
import traceback

sys.path.insert(0, r"c:/Users/YOURAJ/Downloads/IOT Project/Hybrid-LSTM-AQI-Forcasting")

try:
    import main

    result = main.run_lstm_prediction("Delhi")
    print("STATUS", result.get("status"))
    print("HAS_PREDICTIONS", "predictions" in result)
    print("P24", result.get("predictions", {}).get("24h"))
except Exception as exc:
    print("ERR", exc)
    traceback.print_exc()
