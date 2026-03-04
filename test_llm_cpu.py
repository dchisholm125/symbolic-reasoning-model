import time
import sys
import threading
from framework.srm.llm_backend import LLMBackend

def heartbeat():
    i = 0
    while True:
        sys.stdout.write(f'\r[Heartbeat] Computing... {i}s')
        sys.stdout.flush()
        time.sleep(1)
        i += 1

print("Loading LLM backend...")
llm = LLMBackend()
print("\nBackend loaded!")

t = threading.Thread(target=heartbeat, daemon=True)
t.start()

print("\nGenerating...")
im_s = '<' + '|im_start|' + '>'
im_e = '<' + '|im_end|' + '>'
res = llm.complete(
    f"{im_s}system\nWarm up.{im_e}\n{im_s}user\nHi{im_e}\n{im_s}assistant\n",
    max_tokens=10,
    stop=[im_e]
)
print(f"\nResult: {res}")

