import "dotenv/config";
import songs from "./songs.js";
import { searchTrack, addTracks } from "./spotify.js";

// Split "TITLE — ARTIST" into usable components
function parseLine(line) {
  const [title, artist] = line.split(" — ").map(x => x.trim());
  return { title, artist };
}

(async () => {
  console.log(`Processing ${songs.length} songs…`);

  for (const line of songs) {
    const { title, artist } = parseLine(line);

    console.log(`Searching: ${title} — ${artist}`);

    const uri = await searchTrack(title, artist);

    if (!uri) {
      console.log(`❌ NOT FOUND: ${title} — ${artist}`);
      continue;
    }

    console.log(`✔ FOUND: ${title} — ${artist} → ${uri}`);

    await addTracks([uri]);
  }

  console.log("✅ Completed playlist import!");
})();
