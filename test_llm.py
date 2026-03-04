import time
import sys
import threading
from framework.srm.llm_backend import LLMBackend

def heartbeat():
    while True:
        sys.stdout.write('.')
        sys.stdout.flush()
        time.sleep(1)

t = threading.Thread(target=heartbeat, daemon=True)
t.start()

print("Loading LLM backend...")
llm = LLMBackend()
print("\nBackend loaded!")

print("Generating...")
im_s = '<' + '|im_start|' + '>'
im_e = '<' + '|im_end|' + '>'
res = llm.complete(
    f"{im_s}system\nWarm up.{im_e}\n{im_s}user\nHi{im_e}\n{im_s}assistant\n",
    max_tokens=10,
    stop=[im_e]
)
print(f"\nResult: {res}")

