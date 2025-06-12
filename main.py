from api.main import app
import os

port = int(os.environ.get("PORT", 5500))
app.run(host="0.0.0.0", port=port)