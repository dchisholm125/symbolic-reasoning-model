import asyncio
import json
import websockets
import base64
import struct
from typing import Dict, Any

class SolanaLiveNerve:
    """
    Translates REAL Solana/Pyth WebSocket data into the SRM Symbolic Array.
    Now attempting to parse the actual Pyth Price Feed layout.
    """
    def __init__(self, rpc_url: str, pyth_account: str):
        self.rpc_url = rpc_url.replace("https://", "wss://")
        self.pyth_account = pyth_account
        self.price = 0.0
        self.prev_price = 0.0
        self.fee = 5000 

    async def stream(self):
        async with websockets.connect(self.rpc_url) as ws:
            sub = {
                "jsonrpc": "2.0", "id": 1, "method": "accountSubscribe",
                "params": [self.pyth_account, {"encoding": "base64", "commitment": "confirmed"}]
            }
            await ws.send(json.dumps(sub))
            print(f"[Sensory] Connected to Mainnet. Parsing Pyth SOL/USD Feed...")
            
            while True:
                msg = await ws.recv()
                data = json.loads(msg)
                
                if "params" in data:
                    try:
                        # Extract the base64 data from the Solana account
                        raw_base64 = data["params"]["result"]["value"]["data"][0]
                        raw_bytes = base64.b64decode(raw_base64)
                        
                        # Pyth Price Feed Layout (Simplified for this version):
                        # The price is usually stored at offset 208 as a 64-bit integer
                        # and the exponent is at offset 216 as a 32-bit integer.
                        price_raw = struct.unpack_from("<q", raw_bytes, 208)[0]
                        exponent = struct.unpack_from("<l", raw_bytes, 216)[0]
                        
                        real_price = price_raw * (10 ** exponent)
                        
                        if real_price > 0:
                            self.prev_price = self.price if self.price > 0 else real_price
                            self.price = round(real_price, 2)
                    except Exception as e:
                        # Fallback if parsing fails - allows the loop to stay alive
                        pass

    def get_symbol(self, position: float, entry: float):
        delta = round(self.price - self.prev_price, 4)
        pnl = round(self.price - entry, 2) if position == 1.0 else 0.0
        return [self.price, self.fee, position, entry, delta, pnl]