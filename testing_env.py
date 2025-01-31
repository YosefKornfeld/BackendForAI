from dotenv import load_dotenv
import os

load_dotenv()

print("PocketBase URL:", os.getenv("POCKETBASE_URL"))
print("API Token:", os.getenv("POCKETBASE_API_TOKEN"))