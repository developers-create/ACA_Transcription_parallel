import os
import time
import requests
import base64
from dotenv import load_dotenv
import subprocess
from pydantic import BaseModel, Field
from typing import Annotated, Literal, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from google.api_core import exceptions
from dotenv import load_dotenv
import json
import pandas as pd
import re
from fuzzywuzzy import fuzz
from sqlalchemy import text
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Numeric,BigInteger
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import random


load_dotenv()
GOOGLE_API_KEYS = [
    os.getenv(f'GOOGLE_API_KEY_{i}') or os.getenv('GOOGLE_API_KEY') 
    for i in range(1, 21)
]
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

LOGIN_URL = "https://samwad.iotcom.io/api/applogin"
CALLDATA_URL = "https://samwad.iotcom.io/callData"
BASE_AUDIO_URL = "https://samwad.iotcom.io/recording/recording"

USERNAME = "srkh"
PASSWORD = "RahulSR@789"

DATABASE_URL = "mssql+pymssql://sa:y8GPm7unIEoMqpU@node233738-env-3218117.in1.bitss.cloud:1433/Auto_Call_Auditor"
TEMP_DIR = "input_file"
os.makedirs(TEMP_DIR, exist_ok=True)
SLEEP_BETWEEN_CALLS = 10  

_DOCTOR_DF_CACHE = None
_DB_CONNECTION_FAILED = False

progress_lock = Lock()
processed_count = 0
total_files = 0

def get_auth_token():
    token=None
    response = requests.post(LOGIN_URL, json={"username": USERNAME, "password": PASSWORD})
    if response.status_code == 200:
        token=response.json().get("token")
        print(response.json().get("token"))
    return token

def fetch_call_data(token,start_date,end_date):
    headers = {"Authorization": f"Bearer {token}"}
    body = {"startdate": start_date, "enddate": end_date}
    
    response = requests.post(CALLDATA_URL, json=body, headers=headers)
    if response.status_code == 200:
        data = response.json()
        call_list = data.get("result") or data.get("data") or []
        return call_list
    return []

def filter_calls_with_bridge_id(call_list):
    return [item for item in call_list if 'bridgeID' in item]

def download_audio_files(call_list, token):
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    downloaded_files = []
    call_ids = [call.get('bridgeID') for call in call_list if call.get('bridgeID')]
    
    print(f"\n{'='*60}")
    print(f"Total call recordings available: {len(call_ids)}")
    print(f"Starting audio download for all {len(call_ids)} calls")
    print(f"{'='*60}\n")
    
    for idx, callid in enumerate(call_ids):
        url = f"{BASE_AUDIO_URL}{callid}.wav"
        local_path = os.path.join(TEMP_DIR, f"{callid}.wav")
        headers = {"Authorization": f"Bearer {token}"}

        try:
            print(f"⬇ [{idx+1}/{len(call_ids)}] Downloading: {callid}.wav")
            response = requests.get(url, headers=headers, stream=True)
            
            if response.status_code == 200:
                with open(local_path, "wb") as f:
                    f.write(response.content)
                
                file_size = os.path.getsize(local_path) / (1024 * 1024)
                print(f"  ✓ Downloaded: {file_size:.2f} MB")
                downloaded_files.append(local_path)
            else:
                print(f"  ✗ Failed: Status {response.status_code}")
                if response.status_code == 401:
                    print('  ✗ Unauthorized token')
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
        
        time.sleep(0.5)
    
    print(f"\n{'='*60}")
    print(f"Download complete: {len(downloaded_files)}/{len(call_ids)} files")
    print(f"{'='*60}\n")
    
    return downloaded_files

def enhance_audio(input_path):
    
    enhanced_path = input_path.replace(".wav", "_clean.wav")
    enhanced_path = enhanced_path.replace("./input_file", "./enhanced_audio")
    
    os.makedirs("./enhanced_audio", exist_ok=True)
    
    if os.path.exists(enhanced_path):
        return enhanced_path

    print(f"   ✨ Enhancing Audio...")
    
    try:
        command = [
            "ffmpeg", "-y", "-i", input_path,
            "-af", "highpass=f=200, lowpass=f=3000, dynaudnorm", 
            "-ar", "16000", 
            "-ac", "1",     
            enhanced_path
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return enhanced_path
        
    except FileNotFoundError:
        print("FFMPEG not installed.")
        return input_path
    except Exception as e:
        print(f"{e}. Using original.")
        return input_path

class Line(BaseModel):
    text: Annotated[str, Field(..., description="What line did the speaker say")]
    speaker: Annotated[Literal["Agent", "Customer"], Field(..., description="Who said this line")]
    start_time: Annotated[str, Field(..., description="time in MM:SS format")]
    end_time: Annotated[str, Field(..., description="time in MM:SS format")]

class Convo(BaseModel):
    conversation: Annotated[List[Line], Field(..., description="Full conversation")]

def process_single_audio_file(file_path, api_key, file_index):
    global processed_count, total_files
    
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite", 
            api_key=api_key,
            temperature=0
        )
        structured_llm = llm.with_structured_output(Convo)
        
        prompt="Extract the text from this audio file and convert it into agent customer conversation , along with start time stamp and end time stamp , the output should be strictly in hinglish not hindi,carefully analyse the call and separate the speakers sometimes agent may spea first sometimes customer may speak first so be carefull,The hospital name maybe SRK or SR Kalla Hospital detect this initially if something matches like this ('eg:- Namaskar SR Kalla Hospital se baat kar rhi hoon'), and intially they are strictly talking about this hospital not any hostel or other thing , carefully analyse this thing too, but dont add from yorself only if the tone matches then only add, some may start with this opening somemay not , but transcript every call given to you "
        
        with open(file_path, "rb") as f:
            audio_data = base64.b64encode(f.read()).decode("utf-8")
            
        mime_type = "audio/wav" if file_path.endswith('.wav') else "audio/mp3"
        
        response = structured_llm.invoke([
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "media", 
                        "mime_type": mime_type, 
                        "data": audio_data
                    }
                ]
            )
        ])
        
        response_dict = response.model_dump()
        call_id = os.path.basename(file_path).split('.')[0]
        response_dict['call_id'] = call_id
        
        with progress_lock:
            processed_count += 1
            print(f"👍 Processed [{processed_count}/{total_files}]: {call_id}")
        
        return response_dict
        
    except Exception as e:
        if "429" in str(e) or "ResourceExhausted" in str(e):
            return "KEY_EXHAUSTED"
            
        print(f"X Error processing {os.path.basename(file_path)}: {str(e)}")
        return None

def process_audio_files_parallel():
    global processed_count, total_files
    processed_count = 0
    
    files = [f for f in os.listdir('./input_file') if f.endswith('.wav') or f.endswith('.mp3')]
    total_files = len(files)
    data = []

    print(f"Starting Sequential Processing with Dynamic Key Switching...")

    current_key_idx = 0 # EDITED: Track the active key index across all files
    for idx, file in enumerate(files):
        file_path = f"./input_file/{file}"
        
        while True: # EDITED: Retry loop for the specific file if a key is exhausted
            api_key = GOOGLE_API_KEYS[current_key_idx]
            result = process_single_audio_file(file_path, api_key, idx)
            
            if result == "KEY_EXHAUSTED":
                current_key_idx = (current_key_idx + 1) % len(GOOGLE_API_KEYS)
                print(f"🔄 Key Exhausted. Switching to next key: ...{api_key[-5:]}")
                time.sleep(2) # Short buffer before retrying
                continue
            
            if result:
                data.append(result)
            
            print("Sleeping!!!!!!\n")
            time.sleep(4) 
            break 

    return data

def save_output_json(data):
    os.makedirs("./processed_reports", exist_ok=True)
    with open(f"./processed_reports/output.json","w") as f:
        json.dump(data,f)

def extract_agent_conversations(data):
    for item in data:
        convo=""
        for line in item['conversation']:
            if(line['speaker']=='Agent'):
                convo+=f"{line['speaker']}: {line['text']} \n "
        item['agent_convo']=convo

def convert_conversations_to_text(data):
    for item in data:
        convo=""
        for line in item['conversation']:
            convo+=f"{line['speaker']}: {line['text']} \n "
        item['conversation']=convo

def evaluate_bank_info(conversation_data, engine):
    
    global _DOCTOR_DF_CACHE, _DB_CONNECTION_FAILED

    if _DB_CONNECTION_FAILED:
        return 0, "Bank Info Skipped: Database unreachable"

    if isinstance(conversation_data, list):
        transcript = " ".join([
            str(line.get('text', '') if isinstance(line, dict) else getattr(line, 'text', '')) 
            for line in conversation_data
        ]).lower()
    else:
        transcript = str(conversation_data).lower()

    deferral_phrases = ["confirm karke", "check karke", "pata karke", "call back", "wait kijiye"]
    if any(phrase in transcript for phrase in deferral_phrases):
        return 0, "Bank Info: Agent deferred to check details"

    if _DOCTOR_DF_CACHE is None:
        try:
            print("Fetching Doctor Schedule from DB")
            with engine.connect() as conn:
                _DOCTOR_DF_CACHE = pd.read_sql(text("SELECT * FROM DoctorSchedule"), conn)
            
            _DOCTOR_DF_CACHE['clean_name'] = _DOCTOR_DF_CACHE['DoctorName'].astype(str).str.lower().str.replace(r'dr\.?\s*', '', regex=True).str.strip()
            print("Doctor Schedule Cached.")
        except Exception as e:
            print(f"DB Connection Failed.")
            _DB_CONNECTION_FAILED = True
            return 0, "Bank Info: DB Connection Failed"

    df = _DOCTOR_DF_CACHE
    if df.empty: return 0, "Bank Info: DB Empty"

    best_match_row = None
    
    for index, row in df.iterrows():
        full_name = row['clean_name']
        first_name = full_name.split()[0]
        
        if re.search(r'\b' + re.escape(first_name) + r'\b', transcript):
            best_match_row = row
            break 

    if best_match_row is None:
        best_score = 0
        for index, row in df.iterrows():
            score = fuzz.partial_ratio(row['clean_name'], transcript)
            if score > 85 and score > best_score:
                best_score = score
                best_match_row = row
        
        if best_match_row is None:
            return 0, "Bank Info: No registered doctor mentioned"

    db_time_str = str(best_match_row['OPD_Timings']).lower()
    
    transcript_has_digits = bool(re.search(r'\d+', transcript))

    if not transcript_has_digits:
        print(f"Bank Info Match: {best_match_row['DoctorName']} (No time mentioned)")
        return 33, ""

    else:
        db_time_match = re.search(r'\d+', db_time_str)
        
        if db_time_match:
            start_hour = str(int(db_time_match.group()))
            
            if start_hour in transcript:
                print(f"✅ Bank Info Match: {best_match_row['DoctorName']} (Time Verified: {start_hour})")
                return 33, ""
            else:
                print(f"Bank Info Mismatch: {best_match_row['DoctorName']} starts at {start_hour}, but not found in text.")
                return 0, f"Bank Info Training: Wrong Time provided for {best_match_row['DoctorName']}"
        else:
            return 33, ""
        
def calculate_call_score(conversation_text):
    if not conversation_text:
        conversation_text = ""
        
    all_lines = str(conversation_text).split('\n')
    agent_lines = [line for line in all_lines if "agent" in line.lower()]
    last_transcript=" ".join(agent_lines[-10:]).lower()
    agent_transcript = " ".join(agent_lines).lower()

    greeting_score = 0
    end_call_score = 0
    training_needed = []

    greetings = ['good morning', 'good evening', 'namaste', 'namaskar']
    
    help_offer = [
        'how may i help', 'can i help', 'help you', 
        'kya seva', 'kaise madad', 'kya help', 'bataiye', 'kis tarah' 
    ]

    closing_phrases = [
        "anything else", "further assistance", "happy to help", "other help",
        "aur kuch", "aur koi seva", "koi aur sawaal", "aur koi madad", 
        "sewa ka mauka", "dhanyavad","dhanyawad", "thank you", "thanks","Aapka samay dene ke liye dhanyawad","dhanyawaad","dhanyavaad"
    ]

    has_greet = any(x in agent_transcript for x in greetings)
    has_help  = any(x in agent_transcript for x in help_offer)

    if has_greet:
        greeting_score = 33
    else:
        missing = []
        if not has_greet: missing.append("Greeting")
        if not has_help: missing.append("Help Offer")
        training_needed.append(f"Opening Missing: {', '.join(missing)}")

    has_closing = any(x in last_transcript for x in closing_phrases)

    if has_closing:
        end_call_score = 33
    else:
        training_needed.append("Closing Missing: No assistance offer found")

    final_training = " | ".join(training_needed) if training_needed else "Training Not Required"

    return greeting_score, end_call_score, final_training

def setup_database():
    DATABASE_URL = (
        "mssql+pymssql://sa:y8GPm7unIEoMqpU@"
        "node233738-env-3218117.in1.bitss.cloud:1433/Auto_Call_Auditor"
    )

    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    Base = declarative_base()
    
    return engine, SessionLocal, Base

def calculate_scores(data, engine):
    remarks=[]
    
    for item in data:
        greeting_score,end_call_score,training=calculate_call_score(item['agent_convo'])
        bank_info_score,reason=evaluate_bank_info(item['conversation'],engine)
        item['remark']=training+" | "+reason
        item['behaviour_score']=str(greeting_score+end_call_score+bank_info_score)
        item['greeting_score']=str(greeting_score)
        item['bank_info_score']=str(bank_info_score)
        item['end_call_score']=str(end_call_score)

def define_models(Base):
    class Call_Details(Base):
        __tablename__ = "Call_Details"

        id = Column(Integer, primary_key=True, index=True, autoincrement=True)
        hangupcause = Column(String(255), nullable=True)
        hanguptime = Column(BigInteger, nullable=True)
        anstime = Column(BigInteger, nullable=True)
        bridgeID = Column(String(100), unique=True, index=True)
        agent = Column(String(100), nullable=True)
        agentchannel = Column(String(100), nullable=True)
        callType = Column(String(50), nullable=True)
        dialNumber = Column(String(50), nullable=True)
        channelID = Column(String(100), nullable=True) 
        startTime = Column(BigInteger, nullable=True)
        Caller = Column(String(50), nullable=True)
        adminuser = Column(String(50), nullable=True)
        campaign = Column(Integer, nullable=True)    
        playbackstarted = Column(String(255), nullable=True)
        playbackstarttime = Column(BigInteger, nullable=True)
        
        Disposition = Column(String(50), nullable=True)
        Type = Column(String(50), nullable=True)
        isDialed = Column(Integer, nullable=True)  

    class TranscriptionRecord(Base):
        __tablename__ = "TranscriptRecord"

        id = Column(Integer, primary_key=True, index=True)
        caller_Id = Column(String(100), index=True)
        Transcription = Column(Text, nullable=True)
        transcripted_date = Column(DateTime, default=datetime.utcnow)
        BankInfoPcnt=Column(Text,nullable=True)
        BehaviorScorePercent=Column(Text,nullable=True)
        GreetingPcnt=Column(Text,nullable=True)
        End_Call_Pcnt=Column(Text,nullable=True)
        remark = Column(Text, nullable=True)
        training_module = Column(Text, nullable=True)
        missing_details = Column(Text, nullable=True)
        Hold_Time = Column(Numeric(10, 2), nullable=True)
    
    return Call_Details, TranscriptionRecord

def insert_call_details(db, call_with_call_id, Call_Details):
    new_count=0
    seen_ids_in_batch = set()
    skipped_count = 0
    
    print("Adding in Call Details\n")
    for item in call_with_call_id:
                bridge_id = item.get('bridgeID')
                
                if not bridge_id:
                    continue

                if bridge_id in seen_ids_in_batch:
                    skipped_count += 1
                    continue

                exists = db.query(Call_Details).filter(Call_Details.bridgeID == bridge_id).first()
                if exists:
                    skipped_count += 1
                    continue

                seen_ids_in_batch.add(bridge_id)

                record = Call_Details(
                    bridgeID = bridge_id,
                    hangupcause = item.get('hangupcause'),
                    hanguptime = item.get('hanguptime'),
                    agent = item.get('agent'),
                    anstime = item.get('anstime'),
                    campaign = item.get('campaign'),
                    
                    callType = item.get('Type'),
                    
                    agentchannel = str(item.get('agentchannel')) if item.get('agentchannel') is not None else None,
                    dialNumber = item.get('dialNumber') if item.get('dialNumber')is not None else None,
                    channelID = str(item.get('channelID')) if item.get('channelID') is not None else None,
                    startTime = item.get('startTime'),
                    Caller = item.get('Caller'),
                    adminuser = item.get('adminuser'),
                    Disposition = item.get('Disposition'),
                    
                    playbackstarted = str(item.get('playbackstarted')) if item.get('playbackstarted') is not None else None,
                    playbackstarttime = item.get('playbackstarttime') if item.get('playbackstarttime') is not None else None,
                    isDialed = item.get('isDialed') if item.get('isDialed') is not None else None,
                    
               
                )
                
                db.add(record)
                new_count += 1
                print(f"added id:{bridge_id}")
            
    db.commit()

def insert_transcription_records(db, data, TranscriptionRecord):
    print("Adding in Transcription Record")
    for item in data:
        record = TranscriptionRecord(
                caller_Id=item['call_id'],
                Transcription=item['conversation'],
                transcripted_date=datetime.utcnow(),
                BankInfoPcnt=item['bank_info_score'],
                BehaviorScorePercent=item['behaviour_score'],
                GreetingPcnt=item['greeting_score'],
                End_Call_Pcnt=item['end_call_score'],

                Hold_Time=0,
                remark=item['remark'],
                training_module="",
                missing_details=""
            )
        db.add(record)
        
        print(f"{item['call_id']} Saved to Database.")

    db.commit()

def main(process_date=None):
    if process_date is None:
        yesterday = datetime.now() - timedelta(days=1)
        process_date = yesterday.strftime('%Y-%m-%d')
    
    print(f"\n{'='*60}")
    print(f"Processing calls for date: {process_date}")
    print(f"{'='*60}\n")
    
    token = get_auth_token()
    call_list = fetch_call_data(token, process_date, process_date)
    call_with_call_id = filter_calls_with_bridge_id(call_list)
    
    print(f"Found {len(call_with_call_id)} calls with bridge IDs")
    
    if len(call_with_call_id) == 0:
        print("No calls found for processing. Exiting.")
        return
    
    downloaded_files = download_audio_files(call_with_call_id, token)
    
    if len(downloaded_files) == 0:
        print("No audio files downloaded. Exiting.")
        return
    
    data = process_audio_files_parallel()
    
    if len(data) == 0:
        print("No files processed successfully. Exiting.")
        return
    
    save_output_json(data)
    
    extract_agent_conversations(data)
    convert_conversations_to_text(data)
    
    engine, SessionLocal, Base = setup_database()
    calculate_scores(data, engine)
    
    Call_Details, TranscriptionRecord = define_models(Base)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    
    insert_call_details(db, call_with_call_id, Call_Details)
    insert_transcription_records(db, data, TranscriptionRecord)
    
    print(f"\n{'='*60}")
    print(f"✓ Processing complete!")
    print(f"✓ Total calls found: {len(call_with_call_id)}")
    print(f"✓ Audio files downloaded: {len(downloaded_files)}")
    print(f"✓ Calls processed: {len(data)}")
    print(f"✓ Records saved to database: {len(data)}")
    print(f"{'='*60}\n")

if __name__ == "__main__":

    main()








