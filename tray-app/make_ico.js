// Builds a crisp multi-size Windows .ico (16, 48, 128 PNG-in-ICO) from extension/icons
const fs = require('fs');
const path = require('path');

const iconDir = path.resolve(__dirname, '../extension/icons');
const sizes = [16, 48, 128];
const pngs = sizes.map((s) => ({
  size: s,
  data: fs.readFileSync(path.join(iconDir, `icon${s}.png`)),
}));

const header = Buffer.alloc(6);
header.writeUInt16LE(0, 0); // reserved
header.writeUInt16LE(1, 2); // ICO type
header.writeUInt16LE(pngs.length, 4);

const entries = [];
let offset = 6 + pngs.length * 16;

for (const img of pngs) {
  const entry = Buffer.alloc(16);
  entry[0] = img.size >= 256 ? 0 : img.size;
  entry[1] = img.size >= 256 ? 0 : img.size;
  entry[2] = 0;
  entry[3] = 0;
  entry.writeUInt16LE(1, 4); // color planes
  entry.writeUInt16LE(32, 6); // bits per pixel
  entry.writeUInt32LE(img.data.length, 8);
  entry.writeUInt32LE(offset, 12);
  entries.push(entry);
  offset += img.data.length;
}

const ico = Buffer.concat([header, ...entries, ...pngs.map((p) => p.data)]);
const outPath = path.join(__dirname, 'app.ico');
fs.writeFileSync(outPath, ico);
console.log('Generated ICO:', outPath, ico.length, 'bytes');
