"use client";

import React, { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";

type Level = 1 | 2 | 3;
type Screen = "practice" | "learn" | "done";
type Feedback = "idle" | "correct" | "incorrect";

type WordItem = {
  word: string;
  syllables: string[];
  tip: string;
  example: string;
  pattern: string;
};

type SavedProgress = {
  reviewQueue: string[];
  attempts: number;
  firstTryCorrect: number;
  mastered: number;
};

const WORD_BANK: Record<Level, WordItem[]> = {
  1: [
    { word: "the", syllables: ["the"], tip: "Remember the letters t-h-e.", example: "The dog ran home.", pattern: "small word" },
    { word: "was", syllables: ["was"], tip: "It sounds like ‘wuz’ but is spelled w-a-s.", example: "He was very kind.", pattern: "small word" },
    { word: "his", syllables: ["his"], tip: "Finish with s: h-i-s.", example: "His bag is blue.", pattern: "final sound" },
    { word: "from", syllables: ["from"], tip: "Hear both starting sounds: f-r.", example: "She came from school.", pattern: "consonant blend" },
    { word: "with", syllables: ["with"], tip: "Start with w and finish with th.", example: "I went with Dad.", pattern: "small word" },
    { word: "went", syllables: ["went"], tip: "Tap four sounds: w-e-n-t.", example: "We went to class.", pattern: "sound order" },
    { word: "best", syllables: ["best"], tip: "Do not leave out the final t.", example: "Try your best today.", pattern: "final consonant" },
    { word: "help", syllables: ["help"], tip: "Listen for the final l-p blend.", example: "Please help me.", pattern: "final blend" },
    { word: "then", syllables: ["then"], tip: "It begins with th and ends with en.", example: "Finish, then check.", pattern: "small word" },
    { word: "were", syllables: ["were"], tip: "Remember: w-e-r-e.", example: "They were at home.", pattern: "memory word" },
  ],
  2: [
    { word: "made", syllables: ["made"], tip: "Silent e makes a say its name.", example: "She made a card.", pattern: "silent e" },
    { word: "smile", syllables: ["smile"], tip: "Silent e makes the i long.", example: "His smile was bright.", pattern: "silent e" },
    { word: "became", syllables: ["be", "came"], tip: "Build it from be + came.", example: "The sky became dark.", pattern: "word chunks" },
    { word: "letter", syllables: ["let", "ter"], tip: "Use two t’s in the middle.", example: "I wrote a letter.", pattern: "double consonant" },
    { word: "smelled", syllables: ["smelled"], tip: "Start with smell, then add ed.", example: "The flower smelled sweet.", pattern: "word ending" },
    { word: "fluttered", syllables: ["flut", "tered"], tip: "Build it: flutter + ed.", example: "The flag fluttered softly.", pattern: "word ending" },
    { word: "noise", syllables: ["noise"], tip: "The letters oi make the ‘oy’ sound.", example: "I heard a loud noise.", pattern: "vowel team" },
    { word: "their", syllables: ["their"], tip: "Their means something belongs to them.", example: "Their house is nearby.", pattern: "confused word" },
    { word: "inside", syllables: ["in", "side"], tip: "Join the two words in + side.", example: "The book is inside.", pattern: "compound chunks" },
    { word: "started", syllables: ["start", "ed"], tip: "Spell start first, then add ed.", example: "The game started early.", pattern: "word ending" },
  ],
  3: [
    { word: "advantage", syllables: ["ad", "van", "tage"], tip: "Type one chunk at a time: ad-van-tage.", example: "Practice gives an advantage.", pattern: "long word" },
    { word: "exceptional", syllables: ["ex", "cep", "tion", "al"], tip: "Build it in four clear chunks.", example: "She did an exceptional job.", pattern: "long word" },
    { word: "different", syllables: ["dif", "fer", "ent"], tip: "Use double f and finish with ent.", example: "Each person is different.", pattern: "double consonant" },
    { word: "important", syllables: ["im", "por", "tant"], tip: "Listen for three beats: im-por-tant.", example: "Reading is important.", pattern: "long word" },
    { word: "beautiful", syllables: ["beau", "ti", "ful"], tip: "Remember the chunks beau-ti-ful.", example: "The garden looked beautiful.", pattern: "vowel pattern" },
    { word: "remember", syllables: ["re", "mem", "ber"], tip: "Say and type every chunk slowly.", example: "Remember to check your work.", pattern: "long word" },
    { word: "because", syllables: ["be", "cause"], tip: "Big Elephants Can Always Understand Small Elephants.", example: "I smiled because I won.", pattern: "memory word" },
    { word: "sentence", syllables: ["sen", "tence"], tip: "It starts with sen and ends with tence.", example: "Write one clear sentence.", pattern: "word ending" },
    { word: "especially", syllables: ["es", "pe", "cial", "ly"], tip: "Do not add an x after the first e.", example: "I especially like soccer.", pattern: "long word" },
    { word: "experience", syllables: ["ex", "pe", "ri", "ence"], tip: "Build it slowly in four chunks.", example: "It was a good experience.", pattern: "long word" },
  ],
};

const STORAGE_KEY = "spelling-trainer-progress-v3";
const REQUIRED_REPEATS = 2;

function normalise(value: string) {
  return value.trim().toLowerCase();
}

function findWord(word: string): WordItem | undefined {
  return (Object.values(WORD_BANK).flat() as WordItem[]).find((item) => item.word === word);
}

function chooseRandom<T>(items: T[], avoid?: T): T {
  const choices = items.length > 1 ? items.filter((item) => item !== avoid) : items;
  return choices[Math.floor(Math.random() * choices.length)];
}

export default function SpellingTrainer() {
  const [level, setLevel] = useState<Level>(1);
  const [screen, setScreen] = useState<Screen>("practice");
  const [current, setCurrent] = useState<WordItem>(WORD_BANK[1][0]);
  const [input, setInput] = useState("");
  const [feedback, setFeedback] = useState<Feedback>("idle");
  const [message, setMessage] = useState("Press Listen, then type the word.");
  const [showWord, setShowWord] = useState(false);
  const [repeatCount, setRepeatCount] = useState(0);
  const [reviewQueue, setReviewQueue] = useState<string[]>([]);
  const [attempts, setAttempts] = useState(0);
  const [firstTryCorrect, setFirstTryCorrect] = useState(0);
  const [mastered, setMastered] = useState(0);
  const [ready, setReady] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const accuracy = useMemo(
    () => (attempts ? Math.round((firstTryCorrect / attempts) * 100) : 0),
    [attempts, firstTryCorrect],
  );

  const speak = useCallback((text: string, rate = 0.66) => {
    if (typeof window === "undefined" || !("speechSynthesis" in window) || !text) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = rate;
    utterance.pitch = 1;
    utterance.volume = 1;
    const voices = window.speechSynthesis.getVoices();
    const englishVoice = voices.find((voice) => voice.lang.toLowerCase().startsWith("en"));
    if (englishVoice) utterance.voice = englishVoice;
    window.speechSynthesis.speak(utterance);
  }, []);

  const spellAloud = useCallback(() => {
    speak(`${current.word}. ${current.word.split("").join(". ")}`, 0.52);
  }, [current.word, speak]);

  const loadNextWord = useCallback((selectedLevel: Level, forceReview = false) => {
    let next: WordItem | undefined;
    if (reviewQueue.length && (forceReview || Math.random() < 0.4)) {
      const reviewWord = chooseRandom(reviewQueue, current.word);
      next = findWord(reviewWord);
    }
    if (!next) next = chooseRandom(WORD_BANK[selectedLevel], current);

    setCurrent(next);
    setInput("");
    setFeedback("idle");
    setShowWord(false);
    setRepeatCount(0);
    setScreen("practice");
    setMessage(reviewQueue.includes(next.word) ? "Review word: listen and remember." : "Press Listen, then type the word.");
    window.setTimeout(() => {
      speak(next.word);
      inputRef.current?.focus();
    }, 200);
  }, [current, reviewQueue, speak]);

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const saved = JSON.parse(raw) as Partial<SavedProgress>;
        setReviewQueue(Array.isArray(saved.reviewQueue) ? saved.reviewQueue.filter((word) => Boolean(findWord(word))) : []);
        setAttempts(Number(saved.attempts) || 0);
        setFirstTryCorrect(Number(saved.firstTryCorrect) || 0);
        setMastered(Number(saved.mastered) || 0);
      }
    } catch {
      // Corrupt browser storage should never stop the app.
    }
    setReady(true);
  }, []);

  useEffect(() => {
    if (!ready) return;
    const progress: SavedProgress = { reviewQueue, attempts, firstTryCorrect, mastered };
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(progress));
  }, [ready, reviewQueue, attempts, firstTryCorrect, mastered]);

  useEffect(() => {
    if (!ready) return;
    loadNextWord(level);
    // Reload only when the learner changes level.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [level, ready]);

  function submitPractice(event: FormEvent) {
    event.preventDefault();
    if (!normalise(input)) {
      setMessage("Please type the word first.");
      return;
    }

    const correct = normalise(input) === current.word;
    setAttempts((value) => value + 1);

    if (correct) {
      setFirstTryCorrect((value) => value + 1);
      setFeedback("correct");
      setMessage("Correct! Excellent careful listening.");
      setReviewQueue((queue) => queue.filter((word) => word !== current.word));
      speak("Correct. Great job.");
      window.setTimeout(() => loadNextWord(level), 850);
      return;
    }

    setFeedback("incorrect");
    setMessage("Good try. Now learn it in small steps.");
    setReviewQueue((queue) => queue.includes(current.word) ? queue : [...queue, current.word]);
    setInput("");
    setRepeatCount(0);
    setShowWord(true);
    setScreen("learn");
    speak(`The word is ${current.word}`);
    window.setTimeout(() => inputRef.current?.focus(), 100);
  }

  function submitLearning(event: FormEvent) {
    event.preventDefault();
    if (!normalise(input)) return;

    if (normalise(input) !== current.word) {
      setFeedback("incorrect");
      setMessage("Almost. Look at each chunk and try again.");
      setInput("");
      speak("Almost. Try again slowly.");
      return;
    }

    const nextCount = repeatCount + 1;
    setRepeatCount(nextCount);
    setInput("");
    setFeedback("correct");

    if (nextCount >= REQUIRED_REPEATS) {
      setMastered((value) => value + 1);
      setReviewQueue((queue) => queue.filter((word) => word !== current.word));
      setScreen("done");
      setMessage("You learned it! It will return later for review.");
      speak("Excellent. You learned the word.");
    } else {
      setMessage("Correct once. Type it one more time.");
      speak("Correct. One more time.");
      window.setTimeout(() => inputRef.current?.focus(), 100);
    }
  }

  function resetProgress() {
    if (!window.confirm("Reset all saved spelling progress?")) return;
    setReviewQueue([]);
    setAttempts(0);
    setFirstTryCorrect(0);
    setMastered(0);
    window.localStorage.removeItem(STORAGE_KEY);
    loadNextWord(level);
  }

  if (!ready) {
    return <main style={styles.page}><div style={styles.card}>Loading spelling practice…</div></main>;
  }

  return (
    <main style={styles.page}>
      <div style={styles.shell}>
        <header style={{ textAlign: "center", marginBottom: 18 }}>
          <h1 style={{ marginBottom: 4, fontSize: 34 }}>Active Spelling Trainer</h1>
          <p style={{ marginTop: 0, color: "#52627a" }}>Listen • retrieve • correct • review</p>
        </header>

        <section style={styles.statsRow} aria-label="Progress">
          <Stat label="First-try accuracy" value={`${accuracy}%`} />
          <Stat label="Words learned" value={String(mastered)} />
          <Stat label="Review words" value={String(reviewQueue.length)} />
        </section>

        <section style={styles.card}>
          <div style={styles.levelRow}>
            <strong>Difficulty</strong>
            {([1, 2, 3] as Level[]).map((item) => (
              <button
                key={item}
                type="button"
                onClick={() => setLevel(item)}
                style={{ ...styles.smallButton, ...(level === item ? styles.activeButton : {}) }}
              >
                Level {item}
              </button>
            ))}
          </div>

          {screen === "practice" && (
            <form onSubmit={submitPractice}>
              <div style={styles.center}>
                <div style={styles.badge}>{reviewQueue.includes(current.word) ? "Review word" : current.pattern}</div>
                <h2 style={{ marginBottom: 8 }}>Listen and spell</h2>
                <p style={styles.instruction}>The word stays hidden so the brain retrieves it from memory.</p>

                <div style={styles.buttonRow}>
                  <button type="button" style={styles.audioButton} onClick={() => speak(current.word)}>🔊 Listen</button>
                  <button type="button" style={styles.secondaryButton} onClick={() => speak(current.example, 0.72)}>💬 Sentence</button>
                </div>
              </div>

              <input
                ref={inputRef}
                value={input}
                onChange={(event) => setInput(event.target.value)}
                autoComplete="off"
                autoCapitalize="none"
                spellCheck={false}
                aria-label="Type the spelling word"
                placeholder="Type the word"
                style={{ ...styles.input, ...(feedback === "incorrect" ? styles.badInput : {}), ...(feedback === "correct" ? styles.goodInput : {}) }}
              />

              <button type="submit" style={styles.primaryButton}>Check spelling</button>
            </form>
          )}

          {screen === "learn" && (
            <form onSubmit={submitLearning}>
              <div style={styles.center}>
                <div style={styles.badge}>Learn in small steps</div>
                <h2 style={{ fontSize: 38, margin: "14px 0 8px", letterSpacing: 2 }}>{showWord ? current.word : "•••••"}</h2>
                <div style={styles.chunks}>{current.syllables.map((chunk) => <span key={chunk}>{chunk}</span>)}</div>
              </div>

              <div style={styles.learningBox}>
                <p><strong>Pattern:</strong> {current.pattern}</p>
                <p><strong>Memory tip:</strong> {current.tip}</p>
                <p><strong>Sentence:</strong> {current.example}</p>
              </div>

              <div style={styles.buttonRow}>
                <button type="button" style={styles.audioButton} onClick={() => speak(current.word)}>🔊 Say word</button>
                <button type="button" style={styles.secondaryButton} onClick={spellAloud}>🔤 Spell slowly</button>
                <button type="button" style={styles.secondaryButton} onClick={() => setShowWord((value) => !value)}>{showWord ? "🙈 Hide word" : "👁 Show word"}</button>
              </div>

              <p style={{ textAlign: "center", fontWeight: 700 }}>Type it correctly twice: {repeatCount}/{REQUIRED_REPEATS}</p>
              <input
                ref={inputRef}
                value={input}
                onChange={(event) => setInput(event.target.value)}
                autoComplete="off"
                autoCapitalize="none"
                spellCheck={false}
                aria-label="Practice the spelling word"
                placeholder="Copy the word carefully"
                style={{ ...styles.input, ...(feedback === "incorrect" ? styles.badInput : {}) }}
              />
              <button type="submit" style={styles.primaryButton}>Check practice</button>
            </form>
          )}

          {screen === "done" && (
            <div style={styles.center}>
              <div style={{ fontSize: 64 }}>⭐</div>
              <h2>Word learned</h2>
              <p style={{ fontSize: 20 }}><strong>{current.word}</strong></p>
              <button type="button" style={styles.primaryButton} onClick={() => loadNextWord(level)}>Next word</button>
            </div>
          )}

          <div role="status" aria-live="polite" style={{ ...styles.message, ...(feedback === "correct" ? styles.goodMessage : {}), ...(feedback === "incorrect" ? styles.badMessage : {}) }}>
            {message}
          </div>

          <div style={styles.footerButtons}>
            <button type="button" style={styles.textButton} onClick={() => loadNextWord(level, true)} disabled={!reviewQueue.length}>Practice a missed word</button>
            <button type="button" style={styles.textButton} onClick={resetProgress}>Reset progress</button>
          </div>
        </section>

        <p style={{ textAlign: "center", color: "#6d7890", fontSize: 14, marginTop: 16 }}>
          Aim for 8–12 minutes per session. Stop before fatigue.
        </p>
      </div>
    </main>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return <div style={styles.stat}><strong style={{ fontSize: 23 }}>{value}</strong><span>{label}</span></div>;
}

const styles: Record<string, React.CSSProperties> = {
  page: { minHeight: "100vh", background: "linear-gradient(180deg,#edf5ff,#ffffff 65%)", color: "#172033", fontFamily: "Arial,Helvetica,sans-serif", padding: "24px 14px 48px" },
  shell: { width: "100%", maxWidth: 760, margin: "0 auto" },
  card: { background: "#fff", border: "1px solid #d9e4f1", borderRadius: 24, boxShadow: "0 14px 38px rgba(38,59,91,.12)", padding: 24 },
  statsRow: { display: "grid", gridTemplateColumns: "repeat(3,minmax(0,1fr))", gap: 10, marginBottom: 14 },
  stat: { background: "#fff", border: "1px solid #dce5f0", borderRadius: 15, padding: 12, display: "flex", flexDirection: "column", textAlign: "center", color: "#42536c", fontSize: 13 },
  levelRow: { display: "flex", flexWrap: "wrap", alignItems: "center", gap: 8, paddingBottom: 16, borderBottom: "1px solid #e6edf5", marginBottom: 20 },
  smallButton: { border: "1px solid #b8c8dc", background: "#fff", color: "#28466d", borderRadius: 999, padding: "8px 12px", cursor: "pointer", fontWeight: 700 },
  activeButton: { background: "#2457d6", color: "#fff", borderColor: "#2457d6" },
  center: { textAlign: "center" },
  badge: { display: "inline-block", background: "#edf3ff", color: "#29528e", borderRadius: 999, padding: "6px 11px", fontSize: 13, fontWeight: 800 },
  instruction: { color: "#5d6b80", maxWidth: 520, margin: "0 auto 18px", lineHeight: 1.5 },
  buttonRow: { display: "flex", flexWrap: "wrap", justifyContent: "center", gap: 9, margin: "14px 0" },
  audioButton: { border: 0, borderRadius: 13, padding: "12px 17px", background: "#e7f0ff", color: "#1e4785", fontSize: 16, fontWeight: 800, cursor: "pointer" },
  secondaryButton: { border: "1px solid #bdcce0", borderRadius: 13, padding: "12px 15px", background: "#fff", color: "#344e72", fontSize: 15, fontWeight: 700, cursor: "pointer" },
  input: { boxSizing: "border-box", width: "100%", border: "2px solid #9eb3cf", borderRadius: 15, padding: "15px 16px", fontSize: 28, textAlign: "center", letterSpacing: 1.5, outline: "none", margin: "13px 0" },
  goodInput: { borderColor: "#2e9b62", background: "#f3fff8" },
  badInput: { borderColor: "#cc4254", background: "#fff7f8" },
  primaryButton: { width: "100%", border: 0, borderRadius: 14, padding: "14px 20px", background: "#2859d6", color: "#fff", fontSize: 18, fontWeight: 900, cursor: "pointer" },
  chunks: { display: "flex", flexWrap: "wrap", justifyContent: "center", gap: 7, marginBottom: 15 },
  learningBox: { background: "#f7faff", border: "1px solid #dce7f5", borderRadius: 15, padding: "10px 16px", lineHeight: 1.5 },
  message: { marginTop: 17, borderRadius: 13, background: "#f2f5f9", color: "#44536a", padding: "12px 14px", textAlign: "center", fontWeight: 700 },
  goodMessage: { background: "#eafaf1", color: "#187348" },
  badMessage: { background: "#fff0f2", color: "#a72d3f" },
  footerButtons: { display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: 8, marginTop: 12 },
  textButton: { border: 0, background: "transparent", color: "#315a93", textDecoration: "underline", cursor: "pointer", padding: 7 },
};
