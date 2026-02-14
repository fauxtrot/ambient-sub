"""
Test script to verify connection to Svelte API.
"""
import asyncio
import aiohttp

async def test_frame_creation():
    """Test creating a frame via Svelte API."""
    url = "http://localhost:5174/api/frame/create"

    payload = {
        "sessionId": 1,
        "frameType": "webcam",
        "imagePath": "test/python_test.jpg",
        "detections": '[{"class": "laptop", "confidence": 0.88}]',
        "reviewed": False,
        "notes": "Python test frame"
    }

    print(f"Testing connection to: {url}")
    print(f"Payload: {payload}")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as response:
                print(f"Status: {response.status}")
                result = await response.json()
                print(f"Response: {result}")

                if response.status == 200:
                    print("\n[SUCCESS] Frame created successfully")
                    return True
                else:
                    print(f"\n[FAILED] Status {response.status}")
                    return False

    except aiohttp.ClientConnectorError as e:
        print(f"\n[CONNECTION ERROR] {e}")
        print("Make sure Svelte dev server is running on port 5174")
        return False
    except Exception as e:
        print(f"\n[ERROR] {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_frame_creation())
