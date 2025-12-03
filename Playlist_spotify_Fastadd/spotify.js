import fetch from "node-fetch";
import { ACCESS_TOKEN, PLAYLIST_ID } from "./config.js";

export async function searchTrack(title, artist) {
  const query = `${title} ${artist}`;

  const url = `https://api.spotify.com/v1/search?q=${encodeURIComponent(query)}&type=track&limit=1`;

  const res = await fetch(url, {
    headers: { Authorization: `Bearer ${ACCESS_TOKEN}` }
  });

  const data = await res.json();
  if (data.error) console.log("API ERROR:", data.error);


  return data?.tracks?.items?.[0]?.uri || null;
}

export async function addTracks(uris) {
  const url = `https://api.spotify.com/v1/playlists/${PLAYLIST_ID}/tracks`;

  return fetch(url, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${ACCESS_TOKEN}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ uris })
  });
}
