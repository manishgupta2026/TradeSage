import asyncio
from telegram import Bot
import os
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

async def get_chat_id():
    bot = Bot(token=TOKEN)
    me = await bot.get_me()
    print(f"Checking for updates from Bot: {me.username}...")
    print("Waiting for you to send a message... (Polling for 30 seconds)")
    
    for i in range(10):
        updates = await bot.get_updates()
        if updates:
            print("\n--- MESSAGES FOUND ---")
            for u in updates:
                if u.message:
                    print(f"User: {u.message.from_user.first_name} | Chat ID: {u.message.chat_id} | Text: {u.message.text}")
                    print(f"\n>>> YOUR CHAT ID IS: {u.message.chat_id} <<<")
                    return u.message.chat_id
        await asyncio.sleep(3)
    
    print("\nNo messages found after polling. Please send 'Hello' to your bot.")

if __name__ == "__main__":
    if not TOKEN:
        print("Error: TELEGRAM_BOT_TOKEN not found in .env")
    else:
        asyncio.run(get_chat_id())
