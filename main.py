
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise SystemExit("No GEMINI_API_KEY / GOOGLE_API_KEY found in .env")


client = genai.Client(api_key=API_KEY)

MODEL = "gemini-1.5-flash-latest"  

prompt = "What's the capital of the USA and write a short paragraph about it?"

try:
   
    response = client.models.generate_content(model=MODEL, contents=prompt)
   
    if hasattr(response, "text") and response.text:
        print("Gemini replied:\n", response.text)
    else:
       
        try:
            cand = response.candidates[0]
            
            if hasattr(cand, "text") and cand.text:
                print("Gemini replied:\n", cand.text)
            elif hasattr(cand, "content"):
                part = cand.content[0]
                if hasattr(part, "text"):
                    print("Gemini replied:\n", part.text)
                else:
                    print("Gemini replied (raw):\n", part)
        except Exception:
            print("Raw response:\n", response)
except Exception as e:
    print("Error calling Gemini:", type(e).__name__, e)
