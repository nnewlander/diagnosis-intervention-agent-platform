from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uvicorn


if __name__ == "__main__":
    port = int(os.getenv("KG_MOCK_PORT", "9003"))
    uvicorn.run("mock_services.kg_service:app", host="127.0.0.1", port=port, reload=False)
