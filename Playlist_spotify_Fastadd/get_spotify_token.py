import base64
import json
import urllib.parse
import requests

# ====== CONFIGURE THESE ======
CLIENT_ID = "86d778e179e8414488518e6f81995526"
CLIENT_SECRET = "b5e4786f336546d080dd00dddaffa6a0"  # paste from Spotify dashboard
REDIRECT_URI = "https://javierpozo.vercel.app/"
SCOPES = "playlist-modify-public playlist-modify-private"
# ==============================


def build_authorize_url():
    params = {
        "client_id": CLIENT_ID,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPES,
    }
    return "https://accounts.spotify.com/authorize?" + urllib.parse.urlencode(params)


def parse_code_from_redirect_url(url: str) -> str:
    """Extract ?code=... from the full redirect URL you paste."""
    parsed = urllib.parse.urlparse(url)
    qs = urllib.parse.parse_qs(parsed.query)
    if "code" not in qs:
        raise ValueError("No 'code' parameter found in URL. Did you paste the full redirect URL?")
    return qs["code"][0]


def get_access_token_from_code(code: str) -> dict:
    """Exchange authorization code for access + refresh tokens."""
    token_url = "https://accounts.spotify.com/api/token"

    auth_header = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode("utf-8")).decode("utf-8")
    headers = {
        "Authorization": f"Basic {auth_header}",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT_URI,
    }

    resp = requests.post(token_url, headers=headers, data=data)
    if resp.status_code != 200:
        raise RuntimeError(f"Token request failed ({resp.status_code}): {resp.text}")

    return resp.json()


def main():
    # 1) Show authorize URL
    authorize_url = build_authorize_url()
    print("STEP 1: Open this URL in your browser, log in to Spotify and accept:")
    print()
    print(authorize_url)
    print()
    print("After accepting, Spotify will redirect you to a URL like:")
    print("  https://example.com/callback?code=...&state=...")
    print()
    redirect_url = input("STEP 2: Paste the FULL redirect URL here and press Enter:\n> ").strip()

    # 2) Extract code
    code = parse_code_from_redirect_url(redirect_url)
    print(f"\nGot authorization code: {code[:10]}...")

    # 3) Exchange code for tokens
    token_data = get_access_token_from_code(code)

    access_token = token_data.get("access_token")
    refresh_token = token_data.get("refresh_token")
    expires_in = token_data.get("expires_in")

    print("\n===== SPOTIFY TOKENS =====")
    print(f"Access token (SPOTIFY_ACCESS_TOKEN):\n{access_token}\n")
    print(f"Refresh token (save it somewhere safe):\n{refresh_token}\n")
    print(f"Expires in (seconds): {expires_in}")
    print("==========================\n")

    # Optionally, save to file for later
    with open("spotify_token.json", "w", encoding="utf-8") as f:
        json.dump(token_data, f, indent=2)
    print("Tokens saved to spotify_token.json")

    print("\nNow update your .env like this:")
    print("SPOTIFY_ACCESS_TOKEN=" + access_token)


if __name__ == "__main__":
    main()
