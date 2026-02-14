"""
Quick script to check if frames are being stored in SpacetimeDB via Svelte API.
"""
import asyncio
import aiohttp

async def check_frames():
    """Query the Svelte API to get frame count."""
    # We'll need to add a query endpoint in Svelte, but for now let's just
    # print instructions
    print("To check if frames are being stored, you can:")
    print("")
    print("1. Open the Svelte UI at http://localhost:5174")
    print("2. Check the browser console for CreateFrame API calls")
    print("3. Or run this SQL query against SpacetimeDB:")
    print("")
    print('   spacetime sql http://localhost:3000/database/ambient-listener "SELECT COUNT(*) FROM frame"')
    print("")
    print("Or if the frame table doesn't exist yet, you need to:")
    print("1. Rebuild the SpacetimeDB module: cd to ambient-listener/server/AmbientListener.SpacetimeDb && spacetime build")
    print("2. Publish it: spacetime publish ambient-listener")
    print("3. Regenerate TypeScript bindings in the Svelte app")

if __name__ == "__main__":
    asyncio.run(check_frames())
