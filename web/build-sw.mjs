// Stamp a build id into the service worker so a deploy actually reaches users.
// A fixed cache name served the old shell from cache indefinitely.
import { readFileSync, writeFileSync } from "node:fs";

const id = process.env.BUILD_ID || String(Date.now());
const src = readFileSync("public/sw.js", "utf8");
writeFileSync("dist/sw.js", src.replaceAll("__BUILD_ID__", id));
console.log("sw.js cache name: predictor-" + id);
