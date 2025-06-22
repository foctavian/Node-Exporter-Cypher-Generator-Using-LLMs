import hashlib
from datetime import datetime, timezone
import base64
import re

def _cleanup_file(filename):
    new_lines = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('#'):
                continue
            new_lines.append(line)
    return '\n'.join(new_lines)

def extract_nodename(filename): # seaches for the nodename to hash it 
    nodename_pattern = re.compile(r'nodename="([^"]+)"')
    data = _cleanup_file(filename).split('\n')
    for line in data:
        if 'nodename="' in line:
            match = nodename_pattern.search(line)
            if match:
                return match.group(1)
            
def is_encoded(filename):
    parts = filename[:-4].split('_')
    
    if len(parts) != 3:
        return False
    b64_nodename, b64_filename, timestamp_str = parts
    try:
        base64.urlsafe_b64decode(b64_nodename.encode())
        base64.urlsafe_b64decode(b64_filename.encode())
        datetime.fromisoformat(timestamp_str)
    except Exception:
        return False
    return True

def encode_file(filename):
    from os import rename, path
    if not is_encoded(filename):
        dr = filename.split('/')[0]
        nodename = extract_nodename(filename)
        enc_nodename = base64.urlsafe_b64encode(nodename.encode()).decode()
        enc_filename = base64.urlsafe_b64encode(filename.encode()).decode()
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d;%H:%M:%S.%f")
        new_filename = path.join(dr,enc_nodename+'_'+enc_filename+'_'+str(timestamp)+'.txt')
        rename(filename, new_filename)
        return new_filename
# use the name of the nodename and the name of the file as it comes and than hash it => unique IP separated by a _
