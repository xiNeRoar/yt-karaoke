// Real-time stereo pitch shifter AudioWorkletProcessor (±6 semitones, tempo preserved)
// Uses dual sine-windowed overlapping delay taps with 4-point Hermite cubic interpolation.

class KaraokePitchProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      {
        name: "semitones",
        defaultValue: 0,
        minValue: -12,
        maxValue: 12,
        automationRate: "k-rate",
      },
    ];
  }

  constructor() {
    super();
    this.bufSize = 16384;
    this.bufMask = this.bufSize - 1;
    this.bufL = new Float32Array(this.bufSize);
    this.bufR = new Float32Array(this.bufSize);
    this.writePos = 0;
    this.windowSize = 2048; // ~42.7ms at 48kHz — sweet spot for vocals & polyphonic music
    this.minDelay = 64;
    this.phase = 0;
    this.currentRatio = 1.0;
    this.port.onmessage = (ev) => {
      if (ev && ev.data && ev.data.type === "reset") {
        this.bufL.fill(0);
        this.bufR.fill(0);
        this.writePos = 0;
        this.phase = 0;
        const semi = typeof ev.data.semitones === "number" ? ev.data.semitones : 0;
        this.currentRatio = Math.pow(2, semi / 12);
      }
    };
  }

  _readHermite(buf, pos) {
    const mask = this.bufMask;
    const i1 = Math.floor(pos);
    const frac = pos - i1;

    const y0 = buf[(i1 - 1) & mask];
    const y1 = buf[i1 & mask];
    const y2 = buf[(i1 + 1) & mask];
    const y3 = buf[(i1 + 2) & mask];

    const c0 = y1;
    const c1 = 0.5 * (y2 - y0);
    const c2 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
    const c3 = 0.5 * (y3 - y0) + 1.5 * (y1 - y2);

    return ((c3 * frac + c2) * frac + c1) * frac + c0;
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    const output = outputs[0];
    if (!input || input.length === 0 || !output || output.length === 0) {
      return true;
    }

    const inL = input[0];
    const inR = input.length > 1 ? input[1] : input[0];
    const outL = output[0];
    const outR = output.length > 1 ? output[1] : null;
    const blockLen = outL.length;

    const semi = parameters.semitones[0];
    const targetRatio = Math.pow(2, semi / 12);

    // Pure bypass if effectively original key
    if (Math.abs(semi) < 0.001 && Math.abs(this.currentRatio - 1.0) < 0.0005) {
      for (let i = 0; i < blockLen; i++) {
        const sL = inL ? inL[i] : 0;
        const sR = inR ? inR[i] : 0;
        this.bufL[this.writePos] = sL;
        this.bufR[this.writePos] = sR;
        this.writePos = (this.writePos + 1) & this.bufMask;
        outL[i] = sL;
        if (outR) outR[i] = sR;
      }
      return true;
    }

    const W = this.windowSize;
    const minD = this.minDelay;
    const bufL = this.bufL;
    const bufR = this.bufR;
    const mask = this.bufMask;

    for (let i = 0; i < blockLen; i++) {
      this.currentRatio += (targetRatio - this.currentRatio) * 0.003;

      const sL = inL ? inL[i] : 0;
      const sR = inR ? inR[i] : 0;

      bufL[this.writePos] = sL;
      bufR[this.writePos] = sR;

      const p0 = this.phase;
      let p1 = p0 + 0.5;
      if (p1 >= 1.0) p1 -= 1.0;

      const w0 = Math.sin(Math.PI * p0);
      const w1 = Math.sin(Math.PI * p1);

      const readPos0 = this.writePos - (minD + p0 * W);
      const readPos1 = this.writePos - (minD + p1 * W);

      const vL = this._readHermite(bufL, readPos0) * w0 + this._readHermite(bufL, readPos1) * w1;
      outL[i] = vL;

      if (outR) {
        const vR = this._readHermite(bufR, readPos0) * w0 + this._readHermite(bufR, readPos1) * w1;
        outR[i] = vR;
      }

      this.phase += (1.0 - this.currentRatio) / W;
      if (this.phase < 0.0) this.phase += 1.0;
      else if (this.phase >= 1.0) this.phase -= 1.0;

      this.writePos = (this.writePos + 1) & mask;
    }

    return true;
  }
}

registerProcessor("karaoke-pitch-processor", KaraokePitchProcessor);
