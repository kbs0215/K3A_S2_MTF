"""Copernicus 토큰 발급 진단 스크립트"""
from dotenv import load_dotenv
import os, requests

load_dotenv()
u = os.getenv("COPERNICUS_USERNAME", "")
p = os.getenv("COPERNICUS_PASSWORD", "")

print(f"User: [{u}]")
print(f"Pass length: {len(p)}")
print(f"Pass has trailing space: {p != p.strip()}")

try:
    r = requests.post(
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
        data={"client_id": "cdse-public", "grant_type": "password", "username": u, "password": p},
        timeout=30,
    )
    print(f"Status: {r.status_code}")
    if r.status_code != 200:
        print(f"Response: {r.text[:500]}")
    else:
        print("토큰 발급 성공!")
except Exception as e:
    print(f"Error: {e}")
