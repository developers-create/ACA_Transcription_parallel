import schedule
import time
from datetime import datetime, timedelta
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from call_transcriptor_parallel import main as run_transcriptor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scheduler.log'),
        logging.StreamHandler()
    ]
)

def job():
    try:
        current_date = datetime.now().strftime('%Y-%m-%d')
        logging.info(f"Starting transcription job for date: {current_date}")
        
        run_transcriptor()
        
        logging.info(f"Successfully completed transcription job for {current_date}")
        
    except Exception as e:
        logging.error(f"Error running transcription job: {str(e)}", exc_info=True)

def run_scheduler():
    schedule.every().day.at("02:00").do(job)
    
    logging.info("Scheduler started. Job will run daily at 02:00 AM")
    logging.info("Press Ctrl+C to stop the scheduler")
    
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    run_scheduler()
