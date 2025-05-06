"""
Test script for the session manager initialization.
"""

import asyncio
from app.services.storage.session_manager import SessionManager

async def test_session_manager():
    print("Creating SessionManager instance...")
    session_manager = SessionManager()
    print(f"SessionManager initialized with TTL: {session_manager.session_ttl}s, cleanup interval: {session_manager.cleanup_interval}s")
    
    print("Starting cleanup task...")
    await session_manager.start_cleanup_task()
    print("Cleanup task started")
    
    # Wait for a moment to ensure the task is running
    await asyncio.sleep(2)
    
    print("Creating test session...")
    session_id = "test-session-1"
    session = await session_manager.get_session(session_id)
    print(f"Session created: {session_id}")
    
    # Wait a bit more to see the task running
    await asyncio.sleep(5)
    
    print("Cleaning up session manager...")
    await session_manager.cleanup()
    print("Cleanup complete")

if __name__ == "__main__":
    asyncio.run(test_session_manager())