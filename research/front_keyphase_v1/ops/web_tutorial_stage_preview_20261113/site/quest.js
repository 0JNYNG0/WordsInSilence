// Ordered quest state. A recognition verdict must pass its own gate before calling accept().
// This preview uses only the explicitly labeled rehearsal control.
export class Quest {
  constructor(words, onComplete = () => {}) { this.words = [...words]; this.onComplete = onComplete; this.reset(); }
  reset() { this.index = 0; this.pending = null; this.completed = false; this.generation = (this.generation || 0) + 1; }
  accept(word, source = 'preview-control') {
    if (this.completed || this.pending || word !== this.words[this.index]) return null;
    const token = Object.freeze({ generation:this.generation, index:this.index, word, source });
    this.pending = token;
    return token;
  }
  settle(token) {
    if (!token || token !== this.pending || token.generation !== this.generation) return false;
    this.pending = null;
    this.index++;
    if (this.index === this.words.length) {
      this.completed = true;
      this.onComplete(Object.freeze({ words:[...this.words], source:token.source }));
    }
    return true;
  }
}
