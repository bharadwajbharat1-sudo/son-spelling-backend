"use client";

import React, { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";

type Level = 1 | 2 | 3;
type Screen = "practice" | "learn" | "success";
type Tab = "practice" | "parent";
type Feedback = "idle" | "correct" | "incorrect";

type WordItem = {
  word: string;
  syllables: string[];
  tip: string;
  example: string;
  pattern: string;
};

type WordProgress = {
  attempts: number;
  correct: number;
  streak: number;
  dueAt: number;
  lastTyped?: string;
};

type SavedData = {
  progress: Record<string, WordProgress>;
  customWords: WordItem[];
  totalAttempts: number;
  firstTryCorrect: number;
  mastered: number;
  sessions: number;
};

const WORD_BANK: Record<Level, WordItem[]> = {
  1: [
    { word: "the", syllables: ["the"], tip: "Say the letters: t-h-e.", example: "The dog ran home.", pattern: "small function word" },
    { word: "was", syllables: ["was"], tip: "It sounds like ‘wuz’, but ends with a-s.", example: "He was very kind.", pattern: "small function word" },
    { word: "were", syllables: ["were"], tip: "Remember the four letters w-e-r-e.", example: "They were at home.", pattern: "confusing small word" },
    { word: "his", syllables: ["his"], tip: "Finish with s: h-i-s.", example: "His bag is blue.", pattern: "final consonant" },
    { word: "has", syllables: ["has"], tip: "This word has an a in the middle.", example: "She has a book.", pattern: "confusing small word" },
    { word: "from", syllables: ["from"], tip: "Hear both opening sounds: f-r.", example: "She came from school.", pattern: "consonant blend" },
    { word: "with", syllables: ["with"], tip: "Start with w and finish with th.", example: "I went with Dad.", pattern: "final digraph" },
    { word: "then", syllables: ["then"], tip: "It begins with th and ends with en.", example: "Finish, then check.", pattern: "small function word" },
    { word: "went", syllables: ["went"], tip: "Tap four sounds: w-e-n-t.", example: "We went to class.", pattern: "sound order" },
    { word: "best", syllables: ["best"], tip: "Do not leave out the final t.", example: "Try your best today.", pattern: "final consonant" },
    { word: "help", syllables: ["help"], tip: "Listen for the final l-p blend.", example: "Please help me.", pattern: "final blend" },
    { word: "over", syllables: ["o", "ver"], tip: "Two beats: o-ver.", example: "The bird flew over us.", pattern: "two syllables" },
  ],
  2: [
    { word: "made", syllables: ["made"], tip: "Silent e makes a say its name.", example: "She made a card.", pattern: "silent e" },
    { word: "smile", syllables: ["smile"], tip: "Silent e makes the i long.", example: "His smile was bright.", pattern: "silent e" },
    { word: "became", syllables: ["be", "came"], tip: "Build it from be + came.", example: "The sky became dark.", pattern: "word chunks" },
    { word: "letter", syllables: ["let", "ter"], tip: "Use two t’s in the middle.", example: "I wrote a letter.", pattern: "double consonant" },
    { word: "smelled", syllables: ["smell", "ed"], tip: "Spell smell first, then add ed.", example: "The flower smelled sweet.", pattern: "base word + ending" },
    { word: "fluttered", syllables: ["flut", "tered"], tip: "Build it: flutter + ed.", example: "The flag fluttered softly.", pattern: "base word + ending" },
    { word: "noise", syllables: ["noise"], tip: "The letters oi make the ‘oy’ sound.", example: "I heard a loud noise.", pattern: "vowel team" },
    { word: "their", syllables: ["their"], tip: "Their means something belongs to them.", example: "Their house is nearby.", pattern: "confusing word" },
    { word: "inside", syllables: ["in", "side"], tip: "Join the chunks in + side.", example: "The book is inside.", pattern: "compound word" },
    { word: "started", syllables: ["start", "ed"], tip: "Spell start first, then add ed.", example: "The game started early.", pattern: "base word + ending" },
    { word: "noses", syllables: ["nos", "es"], tip: "Start with nose, then add s.", example: "The puppies have wet noses.", pattern: "plural ending" },
    { word: "smiled", syllables: ["smile", "ed"], tip: "Keep the silent e idea in smile, then add d.", example: "He smiled at his friend.", pattern: "past-tense ending" },
  ],
  3: [
    { word: "advantage", syllables: ["ad", "van", "tage"], tip: "Type one chunk at a time: ad-van-tage.", example: "Practice gives an advantage.", pattern: "multisyllable word" },
    { word: "exceptionally", syllables: ["ex", "cep", "tion", "al", "ly"], tip: "Build five chunks slowly.", example: "She performed exceptionally well.", pattern: "multisyllable word" },
    { word: "different", syllables: ["dif", "fer", "ent"], tip: "Use double f and finish with ent.", example: "Each person is different.", pattern: "double consonant" },
    { word: "important", syllables: ["im", "por", "tant"], tip: "Listen for three beats: im-por-tant.", example: "Reading is important.", pattern: "multisyllable word" },
    { word: "beautiful", syllables: ["beau", "ti", "ful"], tip: "Remember the chunks beau-ti-ful.", example: "The garden looked beautiful.", pattern: "unusual vowel pattern" },
    { word: "remember", syllables: ["re", "mem", "ber"], tip: "Say and type every chunk slowly.", example: "Remember to check your work.", pattern: "multisyllable word" },
    { word: "because", syllables: ["be", "cause"], tip: "Big Elephants Can Always Understand Small Elephants.", example: "I smiled because I won.", pattern: "memory word" },
    { word: "sentence", syllables: ["sen", "tence"], tip: "It starts with sen and ends with tence.", example: "Write one clear sentence.", pattern: "word ending" },
    { word: "especially", syllables: ["es", "pe", "cial", "ly"], tip: "Do not add an x after the first e.", example: "I especially like soccer.", pattern: "multisyllable word" },
    { word: "experience", syllables: ["ex", "pe", "ri", "ence"], tip: "Build it slowly in four chunks.", example: "It was a good experience.", pattern: "multisyllable word" },
    { word: "adventuring", syllables: ["ad", "ven", "tur", "ing"], tip: "Build adventure first, then add ing.", example: "They enjoy adventuring outdoors.", pattern: "base word + ending" },
    { word: "expedition", syllables: ["ex", "pe", "di", "tion"], tip: "Listen for four clear beats.", example: "The team began an expedition.", pattern: "multisyllable word" },
  ],
};

const STORAGE_KEY = "active-spelling-trainer-v5";
const SESSION_GOAL = 10;
const REQUIRED_REPEATS = 2;
const REVIEW_DELAYS = [0, 2 * 60 * 1000, 24 * 60 * 60 * 1000, 3 * 24 * 60 * 60 * 1000];

const normalise = (text: string) => text.trim().toLowerCase().replace(/[^a-z'-]/g, "");
const allBuiltIns = () => Object.values(WORD_BANK).flat();

function defaultProgress(): WordProgress {
  return { attempts: 0, correct: 0, streak: 0, dueAt: 0 };
}

function editDistance(a: string, b: string): number {
  const matrix = Array.from({ length: a.length + 1 }, () => Array(b.length + 1).fill(0));
  for (let i = 0; i <= a.length; i++) matrix[i][0] = i;
  for (let j = 0; j <= b.length; j++) matrix[0][j] = j;
  for (let i = 1; i <= a.length; i++) {
    for (let j = 1; j <= b.length; j++) {
      matrix[i][j] = a[i - 1] === b[j - 1]
        ? matrix[i - 1][j - 1]
        : Math.min(matrix[i - 1][j], matrix[i][j - 1], matrix[i - 1][j - 1]) + 1;
    }
  }
  return matrix[a.length][b.length];
}

function diagnose(target: string, typed: string): string {
  const answer = normalise(typed);
  if (!answer) return "No answer was entered.";
  if (answer.length < target.length) return "A letter or sound may have been left out.";
  if (answer.length > target.length) return "An extra letter may have been added.";
  if (target[0] !== answer[0]) return "Check the beginning sound.";
  if (target[target.length - 1] !== answer[answer.length - 1]) return "Check the ending sound.";
  if (editDistance(target, answer) === 1) return "Very close—one letter needs attention.";
  return "Say each syllable slowly and rebuild the word in chunks.";
}

function letterComparison(target: string, typed: string) {
  const answer = normalise(typed);
  const length = Math.max(target.length, answer.length);
  return Array.from({ length }, (_, index) => ({
    expected: target[index] ?? "_",
    typed: answer[index] ?? "_",
    correct: target[index] === answer[index],
  }));
}

function makeCustomWord(raw: string): WordItem | null {
  const word = normalise(raw);
  if (!word || word.length < 2) return null;
  return {
    word,
    syllables: [word],
    tip: "Say the word slowly. Tap each sound, then check every letter.",
    example: `Please spell the word ${word}.`,
    pattern: "custom word",
  };
}

export default function Page() {
  const [tab, setTab] = useState<Tab>("practice");
  const [level, setLevel] = useState<Level>(1);
  const [screen, setScreen] = useState<Screen>("practice");
  const [current, setCurrent] = useState<WordItem>(WORD_BANK[1][0]);
  const [input, setInput] = useState("");
  const [lastWrong, setLastWrong] = useState("");
  const [feedback, setFeedback] = useState<Feedback>("idle");
  const [message, setMessage] = useState("Press Listen, then type the word.");
  const [showWord, setShowWord] = useState(false);
  const [repeatCount, setRepeatCount] = useState(0);
  const [progress, setProgress] = useState<Record<string, WordProgress>>({});
  const [customWords, setCustomWords] = useState<WordItem[]>([]);
  const [customInput, setCustomInput] = useState("");
  const [totalAttempts, setTotalAttempts] = useState(0);
  const [firstTryCorrect, setFirstTryCorrect] = useState(0);
  const [mastered, setMastered] = useState(0);
  const [sessions, setSessions] = useState(0);
  const [sessionCount, setSessionCount] = useState(0);
  const [ready, setReady] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const allWords = useMemo(() => [...allBuiltIns(), ...customWords], [customWords]);
  const levelWords = useMemo(() => [...WORD_BANK[level], ...customWords], [level, customWords]);
  const accuracy = totalAttempts ? Math.round((firstTryCorrect / totalAttempts) * 100) : 0;
  const reviewCount = useMemo(() => allWords.filter((item) => {
    const itemProgress = progress[item.word];
    return itemProgress && itemProgress.attempts > 0 && itemProgress.streak < 3;
  }).length, [allWords, progress]);

  const weakPatterns = useMemo(() => {
    const groups: Record<string, { attempts: number; correct: number }> = {};
    allWords.forEach((item) => {
      const p = progress[item.word];
      if (!p?.attempts) return;
      groups[item.pattern] ??= { attempts: 0, correct: 0 };
      groups[item.pattern].attempts += p.attempts;
      groups[item.pattern].correct += p.correct;
    });
    return Object.entries(groups)
      .map(([pattern, data]) => ({ pattern, accuracy: Math.round((data.correct / data.attempts) * 100), attempts: data.attempts }))
      .sort((a, b) => a.accuracy - b.accuracy || b.attempts - a.attempts)
      .slice(0, 5);
  }, [allWords, progress]);

  const speak = useCallback((text: string, rate = 0.65) => {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = rate;
    const voices = window.speechSynthesis.getVoices();
    const voice = voices.find((item) => item.lang.toLowerCase().startsWith("en"));
    if (voice) utterance.voice = voice;
    window.speechSynthesis.speak(utterance);
  }, []);

  const chooseNext = useCallback((forceReview = false) => {
    const now = Date.now();
    const candidates = levelWords.filter((item) => item.word !== current.word);
    const due = candidates.filter((item) => {
      const p = progress[item.word];
      return p && p.attempts > 0 && p.streak < 3 && p.dueAt <= now;
    });
    const weak = candidates.filter((item) => {
      const p = progress[item.word];
      return p && p.attempts > 0 && p.streak < 3;
    });
    let pool = candidates;
    if (due.length && (forceReview || Math.random() < 0.6)) pool = due;
    else if (weak.length && (forceReview || Math.random() < 0.35)) pool = weak;
    const next = pool[Math.floor(Math.random() * pool.length)] ?? levelWords[0];

    setCurrent(next);
    setInput("");
    setLastWrong("");
    setFeedback("idle");
    setShowWord(false);
    setRepeatCount(0);
    setScreen("practice");
    setMessage(progress[next.word]?.attempts ? "Review word: listen carefully and retrieve it." : "Press Listen, then type the word.");
    window.setTimeout(() => {
      speak(next.word);
      inputRef.current?.focus();
    }, 180);
  }, [current.word, levelWords, progress, speak]);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const saved = JSON.parse(raw) as Partial<SavedData>;
        setProgress(saved.progress ?? {});
        setCustomWords(Array.isArray(saved.customWords) ? saved.customWords : []);
        setTotalAttempts(Number(saved.totalAttempts) || 0);
        setFirstTryCorrect(Number(saved.firstTryCorrect) || 0);
        setMastered(Number(saved.mastered) || 0);
        setSessions(Number(saved.sessions) || 0);
      }
    } catch {
      // Bad browser storage must never break the app.
    }
    setReady(true);
  }, []);

  useEffect(() => {
    if (!ready) return;
    const data: SavedData = { progress, customWords, totalAttempts, firstTryCorrect, mastered, sessions };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
  }, [ready, progress, customWords, totalAttempts, firstTryCorrect, mastered, sessions]);

  useEffect(() => {
    if (!ready) return;
    chooseNext(false);
    // Intentionally run when level changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [level, ready]);

  function updateWordProgress(word: string, correct: boolean, typed: string) {
    setProgress((previous) => {
      const old = previous[word] ?? defaultProgress();
      const nextStreak = correct ? Math.min(old.streak + 1, 3) : 0;
      const delay = REVIEW_DELAYS[Math.min(nextStreak, REVIEW_DELAYS.length - 1)];
      return {
        ...previous,
        [word]: {
          attempts: old.attempts + 1,
          correct: old.correct + (correct ? 1 : 0),
          streak: nextStreak,
          dueAt: Date.now() + delay,
          lastTyped: typed,
        },
      };
    });
  }

  function finishAttempt() {
    setSessionCount((count) => {
      const next = count + 1;
      if (next === SESSION_GOAL) setSessions((value) => value + 1);
      return next;
    });
  }

  function submitPractice(event: FormEvent) {
    event.preventDefault();
    const typed = normalise(input);
    if (!typed) {
      setMessage("Type the word first.");
      return;
    }
    const correct = typed === current.word;
    setTotalAttempts((value) => value + 1);
    updateWordProgress(current.word, correct, typed);
    finishAttempt();

    if (correct) {
      setFirstTryCorrect((value) => value + 1);
      setFeedback("correct");
      setMessage("Correct—excellent careful listening!");
      speak("Correct. Great work.");
      window.setTimeout(() => chooseNext(false), 850);
      return;
    }

    setLastWrong(typed);
    setFeedback("incorrect");
    setMessage(diagnose(current.word, typed));
    setInput("");
    setRepeatCount(0);
    setShowWord(true);
    setScreen("learn");
    speak(`The word is ${current.word}`);
  }

  function submitLearning(event: FormEvent) {
    event.preventDefault();
    const typed = normalise(input);
    if (!typed) return;
    if (typed !== current.word) {
      setLastWrong(typed);
      setFeedback("incorrect");
      setMessage(diagnose(current.word, typed));
      setInput("");
      speak("Almost. Try again slowly.");
      return;
    }

    const next = repeatCount + 1;
    setRepeatCount(next);
    setInput("");
    setFeedback("correct");
    if (next >= REQUIRED_REPEATS) {
      setMastered((value) => value + 1);
      setProgress((previous) => ({
        ...previous,
        [current.word]: {
          ...(previous[current.word] ?? defaultProgress()),
          streak: Math.max(previous[current.word]?.streak ?? 0, 1),
          dueAt: Date.now() + REVIEW_DELAYS[1],
        },
      }));
      setScreen("success");
      setMessage("Learned for now. This word will return later.");
      speak("Excellent. You learned the word.");
    } else {
      setMessage("Correct once. Hide it and type it one more time.");
      setShowWord(false);
      speak("Correct. One more time from memory.");
    }
  }

  function addCustomWords() {
    const words = customInput.split(/[\n,]+/).map(makeCustomWord).filter(Boolean) as WordItem[];
    if (!words.length) return;
    setCustomWords((previous) => {
      const map = new Map(previous.map((item) => [item.word, item]));
      words.forEach((item) => map.set(item.word, item));
      return Array.from(map.values());
    });
    setCustomInput("");
  }

  function exportProgress() {
    const payload = {
      exportedAt: new Date().toISOString(),
      summary: { totalAttempts, firstTryCorrect, accuracy, mastered, sessions },
      words: allWords.map((item) => ({ ...item, progress: progress[item.word] ?? defaultProgress() })),
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `spelling-progress-${new Date().toISOString().slice(0, 10)}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  function resetAll() {
    if (!window.confirm("Delete all saved spelling progress and custom words?")) return;
    localStorage.removeItem(STORAGE_KEY);
    setProgress({});
    setCustomWords([]);
    setTotalAttempts(0);
    setFirstTryCorrect(0);
    setMastered(0);
    setSessions(0);
    setSessionCount(0);
    setTab("practice");
    setCurrent(WORD_BANK[1][0]);
  }

  if (!ready) return <main style={styles.page}><section style={styles.card}>Loading…</section></main>;

  return (
    <main style={styles.page}>
      <div style={styles.shell}>
        <header style={styles.header}>
          <div>
            <h1 style={styles.title}>Active Spelling Trainer</h1>
            <p style={styles.subtitle}>Short, calm practice with retrieval and spaced review</p>
          </div>
          <nav style={styles.tabs}>
            <button style={{ ...styles.tab, ...(tab === "practice" ? styles.activeTab : {}) }} onClick={() => setTab("practice")}>Practice</button>
            <button style={{ ...styles.tab, ...(tab === "parent" ? styles.activeTab : {}) }} onClick={() => setTab("parent")}>Parent view</button>
          </nav>
        </header>

        <section style={styles.statsGrid}>
          <Stat label="First-try accuracy" value={`${accuracy}%`} />
          <Stat label="Learned" value={String(mastered)} />
          <Stat label="Review queue" value={String(reviewCount)} />
          <Stat label="Today’s session" value={`${Math.min(sessionCount, SESSION_GOAL)}/${SESSION_GOAL}`} />
        </section>

        {tab === "practice" ? (
          <section style={styles.card}>
            <div style={styles.levelRow}>
              <strong>Difficulty</strong>
              {([1, 2, 3] as Level[]).map((item) => (
                <button key={item} onClick={() => setLevel(item)} style={{ ...styles.pill, ...(level === item ? styles.activePill : {}) }}>Level {item}</button>
              ))}
              <button onClick={() => chooseNext(true)} style={styles.reviewButton}>Review a weak word</button>
            </div>

            {sessionCount >= SESSION_GOAL && (
              <div style={styles.breakBanner}>🎉 Session complete. Ten words are enough for today—stop before fatigue.</div>
            )}

            {screen === "practice" && (
              <form onSubmit={submitPractice}>
                <div style={styles.center}>
                  <span style={styles.badge}>{progress[current.word]?.attempts ? "Spaced review" : current.pattern}</span>
                  <h2>Listen and spell</h2>
                  <p style={styles.muted}>The word stays hidden so memory does the work.</p>
                  <div style={styles.buttonRow}>
                    <button type="button" style={styles.audioButton} onClick={() => speak(current.word)}>🔊 Listen</button>
                    <button type="button" style={styles.secondaryButton} onClick={() => speak(current.example, 0.72)}>💬 Use in sentence</button>
                  </div>
                </div>
                <input ref={inputRef} value={input} onChange={(e) => setInput(e.target.value)} autoComplete="off" autoCapitalize="none" spellCheck={false} placeholder="Type the word" style={styles.input} />
                <button style={styles.primaryButton}>Check spelling</button>
              </form>
            )}

            {screen === "learn" && (
              <form onSubmit={submitLearning}>
                <div style={styles.center}>
                  <span style={styles.badge}>Learn in small steps</span>
                  <h2 style={styles.word}>{showWord ? current.word : "••••••"}</h2>
                  <div style={styles.chunks}>{current.syllables.map((chunk, index) => <span key={`${chunk}-${index}`} style={styles.chunk}>{chunk}</span>)}</div>
                </div>

                {lastWrong && (
                  <div style={styles.comparisonBox}>
                    <strong>Your attempt compared with the word</strong>
                    <div style={styles.letterRow}>
                      {letterComparison(current.word, lastWrong).map((item, index) => (
                        <div key={index} style={{ ...styles.letterCell, ...(item.correct ? styles.correctLetter : styles.wrongLetter) }}>
                          <span>{item.expected}</span><small>{item.typed}</small>
                        </div>
                      ))}
                    </div>
                    <small>Top letter = correct word · bottom letter = typed answer</small>
                  </div>
                )}

                <div style={styles.learningBox}>
                  <p><strong>Pattern:</strong> {current.pattern}</p>
                  <p><strong>Memory tip:</strong> {current.tip}</p>
                  <p><strong>Sentence:</strong> {current.example}</p>
                </div>
                <div style={styles.buttonRow}>
                  <button type="button" style={styles.audioButton} onClick={() => speak(current.word)}>🔊 Say word</button>
                  <button type="button" style={styles.secondaryButton} onClick={() => speak(`${current.word}. ${current.word.split("").join(". ")}`, 0.5)}>🔤 Spell slowly</button>
                  <button type="button" style={styles.secondaryButton} onClick={() => setShowWord((value) => !value)}>{showWord ? "🙈 Hide" : "👁 Show"}</button>
                </div>
                <p style={styles.center}><strong>Type correctly twice: {repeatCount}/{REQUIRED_REPEATS}</strong></p>
                <input ref={inputRef} value={input} onChange={(e) => setInput(e.target.value)} autoComplete="off" autoCapitalize="none" spellCheck={false} placeholder="Type carefully" style={styles.input} />
                <button style={styles.primaryButton}>Check practice</button>
              </form>
            )}

            {screen === "success" && (
              <div style={styles.center}>
                <div style={{ fontSize: 60 }}>⭐</div>
                <h2>Word learned for now</h2>
                <p style={styles.word}>{current.word}</p>
                <p style={styles.muted}>It will return later so the learning becomes stronger.</p>
                <button style={styles.primaryButton} onClick={() => chooseNext(false)}>Next word</button>
              </div>
            )}

            <div role="status" aria-live="polite" style={{ ...styles.message, ...(feedback === "correct" ? styles.goodMessage : {}), ...(feedback === "incorrect" ? styles.badMessage : {}) }}>{message}</div>
          </section>
        ) : (
          <section style={styles.parentGrid}>
            <div style={styles.card}>
              <h2>Personal word list</h2>
              <p style={styles.muted}>Paste school words or words he commonly misspells. Separate them with commas or new lines.</p>
              <textarea value={customInput} onChange={(e) => setCustomInput(e.target.value)} placeholder="friend, school, because" style={styles.textarea} />
              <button style={styles.primaryButton} onClick={addCustomWords}>Add words</button>
              <p><strong>{customWords.length}</strong> custom words saved.</p>
              {customWords.length > 0 && <div style={styles.tagList}>{customWords.map((item) => <span key={item.word} style={styles.tag}>{item.word}</span>)}</div>}
            </div>

            <div style={styles.card}>
              <h2>Weak spelling patterns</h2>
              {weakPatterns.length ? weakPatterns.map((item) => (
                <div key={item.pattern} style={styles.patternRow}>
                  <span>{item.pattern}</span><strong>{item.accuracy}%</strong>
                </div>
              )) : <p style={styles.muted}>Complete a few practice words to see patterns.</p>}
              <p style={styles.tipBox}>Focus on one weak pattern for a week rather than increasing the number of words.</p>
            </div>

            <div style={styles.card}>
              <h2>Most difficult words</h2>
              {allWords
                .filter((item) => progress[item.word]?.attempts)
                .sort((a, b) => {
                  const pa = progress[a.word]; const pb = progress[b.word];
                  return (pa.correct / pa.attempts) - (pb.correct / pb.attempts);
                })
                .slice(0, 8)
                .map((item) => {
                  const p = progress[item.word];
                  return <div key={item.word} style={styles.patternRow}><span>{item.word}<small style={styles.smallNote}> last: {p.lastTyped || "—"}</small></span><strong>{Math.round((p.correct / p.attempts) * 100)}%</strong></div>;
                })}
              {!Object.keys(progress).length && <p style={styles.muted}>No attempts recorded yet.</p>}
            </div>

            <div style={styles.card}>
              <h2>Progress controls</h2>
              <p>Total attempts: <strong>{totalAttempts}</strong></p>
              <p>Completed 10-word sessions: <strong>{sessions}</strong></p>
              <div style={styles.buttonColumn}>
                <button style={styles.secondaryButton} onClick={exportProgress}>Download progress JSON</button>
                <button style={styles.dangerButton} onClick={resetAll}>Reset all data</button>
              </div>
            </div>
          </section>
        )}

        <footer style={styles.footer}>Recommended: 8–12 minutes, 4–5 days per week. Praise effort and strategy, not speed.</footer>
      </div>
    </main>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return <div style={styles.stat}><strong>{value}</strong><span>{label}</span></div>;
}

const styles: Record<string, React.CSSProperties> = {
  page: { minHeight: "100vh", background: "linear-gradient(180deg,#edf5ff,#fff 65%)", color: "#172033", fontFamily: "Arial,Helvetica,sans-serif", padding: "22px 14px 50px" },
  shell: { maxWidth: 900, margin: "0 auto" },
  header: { display: "flex", alignItems: "center", justifyContent: "space-between", gap: 16, flexWrap: "wrap", marginBottom: 16 },
  title: { margin: 0, fontSize: 34 }, subtitle: { margin: "5px 0 0", color: "#5f6f86" },
  tabs: { display: "flex", background: "#dfe9f8", padding: 4, borderRadius: 12 },
  tab: { border: 0, background: "transparent", padding: "10px 14px", borderRadius: 9, cursor: "pointer", fontWeight: 800, color: "#35516f" },
  activeTab: { background: "#fff", color: "#1f4ca5", boxShadow: "0 2px 8px rgba(30,60,100,.14)" },
  statsGrid: { display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(145px,1fr))", gap: 10, marginBottom: 14 },
  stat: { background: "#fff", border: "1px solid #dce5f0", borderRadius: 15, padding: 13, display: "flex", flexDirection: "column", textAlign: "center", color: "#42536c", fontSize: 13 },
  card: { background: "#fff", border: "1px solid #d9e4f1", borderRadius: 22, boxShadow: "0 14px 38px rgba(38,59,91,.10)", padding: 22 },
  parentGrid: { display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(300px,1fr))", gap: 14 },
  levelRow: { display: "flex", flexWrap: "wrap", alignItems: "center", gap: 8, paddingBottom: 16, borderBottom: "1px solid #e6edf5", marginBottom: 20 },
  pill: { border: "1px solid #b8c8dc", background: "#fff", color: "#28466d", borderRadius: 999, padding: "8px 12px", cursor: "pointer", fontWeight: 700 },
  activePill: { background: "#2457d6", color: "#fff", borderColor: "#2457d6" },
  reviewButton: { marginLeft: "auto", border: 0, background: "#eef4ff", color: "#2855a0", borderRadius: 999, padding: "9px 13px", cursor: "pointer", fontWeight: 800 },
  breakBanner: { background: "#fff8dc", border: "1px solid #ead486", borderRadius: 13, padding: 12, textAlign: "center", marginBottom: 16, fontWeight: 700 },
  center: { textAlign: "center" }, muted: { color: "#607086", lineHeight: 1.5 },
  badge: { display: "inline-block", background: "#edf3ff", color: "#29528e", borderRadius: 999, padding: "6px 11px", fontSize: 13, fontWeight: 800 },
  buttonRow: { display: "flex", flexWrap: "wrap", justifyContent: "center", gap: 9, margin: "15px 0" },
  audioButton: { border: 0, borderRadius: 13, padding: "12px 17px", background: "#e7f0ff", color: "#1e4785", fontSize: 16, fontWeight: 800, cursor: "pointer" },
  secondaryButton: { border: "1px solid #bdcce0", borderRadius: 13, padding: "12px 15px", background: "#fff", color: "#344e72", fontSize: 15, fontWeight: 700, cursor: "pointer" },
  primaryButton: { width: "100%", border: 0, borderRadius: 14, padding: "14px 20px", background: "#2859d6", color: "#fff", fontSize: 18, fontWeight: 900, cursor: "pointer" },
  dangerButton: { border: "1px solid #d79099", borderRadius: 13, padding: "12px 15px", background: "#fff5f6", color: "#a52a3b", fontWeight: 800, cursor: "pointer" },
  input: { boxSizing: "border-box", width: "100%", border: "2px solid #9eb3cf", borderRadius: 15, padding: "15px 16px", fontSize: 28, textAlign: "center", letterSpacing: 1.5, outline: "none", margin: "13px 0" },
  textarea: { boxSizing: "border-box", width: "100%", minHeight: 120, resize: "vertical", border: "1px solid #aec0d7", borderRadius: 13, padding: 12, fontSize: 16, marginBottom: 10 },
  word: { fontSize: 37, letterSpacing: 2, margin: "14px 0 8px" },
  chunks: { display: "flex", flexWrap: "wrap", justifyContent: "center", gap: 7, marginBottom: 15 },
  chunk: { background: "#eef4ff", border: "1px solid #cad9ed", borderRadius: 9, padding: "7px 10px", fontSize: 20, fontWeight: 800 },
  learningBox: { background: "#f7faff", border: "1px solid #dce7f5", borderRadius: 15, padding: "10px 16px", lineHeight: 1.5 },
  comparisonBox: { background: "#fff8f0", border: "1px solid #ecd0a9", borderRadius: 15, padding: 14, margin: "14px 0", textAlign: "center" },
  letterRow: { display: "flex", justifyContent: "center", flexWrap: "wrap", gap: 5, margin: "10px 0" },
  letterCell: { width: 36, borderRadius: 8, padding: "5px 2px", display: "flex", flexDirection: "column", fontSize: 20, fontWeight: 900 },
  correctLetter: { background: "#e9f9f0", color: "#20754b" }, wrongLetter: { background: "#ffe9ec", color: "#a92c40" },
  message: { marginTop: 17, borderRadius: 13, background: "#f2f5f9", color: "#44536a", padding: "12px 14px", textAlign: "center", fontWeight: 700 },
  goodMessage: { background: "#eafaf1", color: "#187348" }, badMessage: { background: "#fff0f2", color: "#a72d3f" },
  patternRow: { display: "flex", justifyContent: "space-between", gap: 12, padding: "10px 0", borderBottom: "1px solid #edf0f4" },
  smallNote: { color: "#7a8799", marginLeft: 6 }, tipBox: { background: "#f3f7ff", borderRadius: 12, padding: 12, color: "#425b7c", lineHeight: 1.45 },
  tagList: { display: "flex", flexWrap: "wrap", gap: 7 }, tag: { background: "#eef3fb", borderRadius: 999, padding: "6px 10px", fontSize: 14 },
  buttonColumn: { display: "flex", flexDirection: "column", gap: 9 },
  footer: { textAlign: "center", color: "#66758a", fontSize: 14, marginTop: 17 },
};
