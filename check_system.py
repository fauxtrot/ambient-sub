"""
Quick health check for multi-process executive system.
Shows what's running and recent activity.
"""

import requests
import json

def check_system():
    print("=" * 60)
    print("Multi-Process Executive System Health Check")
    print("=" * 60)

    # Check bridge server
    try:
        # Try to query SpacetimeDB via Svelte API
        response = requests.get("http://localhost:5174/api/sessions", timeout=2)
        print("✓ Svelte server running (port 5174)")
    except:
        print("✗ Svelte server NOT running (port 5174)")

    # Check bridge WebSocket (indirectly via trying to connect)
    try:
        import websocket
        ws = websocket.create_connection("ws://localhost:8175", timeout=2)
        ws.close()
        print("✓ Bridge server running (port 8175)")
    except:
        print("✗ Bridge server NOT running (port 8175)")

    # Check SpacetimeDB
    try:
        response = requests.get("http://127.0.0.1:3000/database/dns/ambient-listener", timeout=2)
        print("✓ SpacetimeDB running (port 3000)")
    except:
        print("✗ SpacetimeDB NOT running (port 3000)")

    # Check Ollama
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        print("✓ Ollama running (port 11434)")
    except:
        print("✗ Ollama NOT running (port 11434)")

    print("\n" + "=" * 60)
    print("Expected Running Processes:")
    print("=" * 60)
    print("Terminal 1: npm run dev:all (Svelte + Bridge)")
    print("Terminal 2: python -m ambient_subconscious.executive.observer")
    print("Terminal 3: python -m ambient_subconscious.executive.llm_process")
    print("Terminal 4: python -m ambient_subconscious.executive.user_input")
    print("=" * 60)

if __name__ == "__main__":
    check_system()
