import os
import asyncio
from dotenv import load_dotenv
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

async def main():

    async with AsyncPostgresSaver.from_conn_string(DATABASE_URL) as checkpointer:
        
        await checkpointer.setup()

        print("Checkpointer is ready!")

if __name__=="__main__":
    asyncio.run(main())
