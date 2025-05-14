import hashlib
from datetime import datetime, timezone
import base64

def hash_string(input_string: str, length: int = 8) -> str:
    """Hashes a string and returns a truncated hash."""
    sha256_hash = hashlib.sha256(input_string.encode()).hexdigest()
    return sha256_hash[:length]

# use the name of the nodename and the name of the file as it comes and than hash it => unique IP separated by a _
print( base64.urlsafe_b64encode("vofr-ThinkPad-E570".encode()).decode())
print(base64.urlsafe_b64encode("node_exporter_metrics".encode()))
print(datetime.now(timezone.utc).isoformat())