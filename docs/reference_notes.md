# Reference notes: for Riley's review

These six short notes are how *Hold Me Tight* and *The Seven Principles for Making Marriage Work* get into the app. The companion can search and read them during a conversation, in the same way it reads your saved agreements.

They are written in our own words from general knowledge of the books' ideas. None of the books' text, exercises or questionnaires are used. **They are already in the app** (`REFERENCE_DOCS` in `app/memory/memory_store.py`). If you read any of the books, check this against what you learn; any corrections go into the app as an update.

## What to check

For each note:

- [ ] **Accurate:** is this a fair summary of the idea? Wrong is worse than missing.
- [ ] **Names and lists:** where a note lists named items (the seven conversations, the Demon Dialogues, the seven principles), are the names and order right? These are marked ⚑.
- [ ] **Our words:** does any sentence sound too close to the book's own wording? If so, mark it and we'll rephrase it.
- [ ] **Tone:** would this help two people feel understood, without either of them feeling blamed?

Mark changes inline, or just tell Jordan what to change.

---

## 1. EFT basics: the relationship as an attachment bond

*Key: `reference:eft-attachment-basics`. Source: Sue Johnson, Hold Me Tight (2008).*

Emotionally Focused Therapy (EFT) sees a committed relationship as an attachment bond, like the bond between a parent and child. It's a basic need for closeness and safety, not a sign of weakness. Johnson sums up what partners need from each other as **A.R.E.**:

- **Accessible:** can I reach you?
- **Responsive:** will you respond when I need you?
- **Engaged:** will you stay close and involved?

Many arguments that seem to be about chores, money or phones carry a deeper question underneath: "Are you there for me? Do I matter to you?" When that question feels unanswered, partners protest (criticise, push) or protect themselves (go quiet, withdraw).

**How the companion uses it:** it looks under the complaint for the attachment need, and reflects it back to both partners ("it sounds like you both want to feel close and on the same side"), rather than refereeing the content of the argument. It draws on these ideas; it never claims to deliver EFT, which is done by trained therapists.

---

## 2. EFT: negative cycles ("Demon Dialogues")

*Key: `reference:eft-negative-cycles`. Source: Sue Johnson, Hold Me Tight (2008).*

Johnson describes recurring negative patterns that couples get stuck in, which she calls Demon Dialogues. ⚑ The three she names are:

- **Find the Bad Guy:** mutual blame; each partner attacks to protect themselves.
- **The Protest Polka:** one partner pursues (pushes, criticises, demands contact), and the other withdraws (goes quiet, shuts down). The most common cycle. Each move triggers more of the other.
- **Freeze and Flee:** both partners withdraw, and the relationship goes cold and distant.

The key idea is that **the cycle is the enemy, not the partner**. Both people are caught in it, and both moves make sense as attempts to protect the bond.

**How the companion uses it:** it names the cycle as something the couple are caught in together ("the more one of you pushes, the more the other pulls back"), never as one partner's fault. This reinforces the app's neutrality rule. It uses the couple's own name for their cycle where they have one.

---

## 3. EFT: the seven conversations

*Key: `reference:eft-seven-conversations`. Source: Sue Johnson, Hold Me Tight (2008).*

The book is organised around seven conversations that build a more secure bond. ⚑ In order:

1. **Recognising the Demon Dialogues:** spotting the negative cycle together.
2. **Finding the raw spots:** the old sensitivities that a partner's words or actions can rub against.
3. **Revisiting a rocky moment:** going back over a fight calmly, to understand what happened between you.
4. **Hold me tight:** reaching for each other and sharing what you need, softly and directly.
5. **Forgiving injuries:** repairing specific moments where one partner felt abandoned or betrayed.
6. **Bonding through sex and touch:** closeness as part of the bond.
7. **Keeping your love alive:** looking after the bond and building a shared story.

**How the companion uses it:** as a map of where a conversation might sit, and to suggest a gentle next step. The app's own conflict-debrief exercise is close in spirit to conversation 3, but it's our own design. The companion does not walk couples through the book's conversations as a programme.

---

## 4. Gottman: the seven principles

*Key: `reference:gottman-seven-principles`. Source: John Gottman and Nan Silver, The Seven Principles for Making Marriage Work (1999).*

⚑ Gottman's seven principles, in our words:

1. **Enhance your love maps:** keep learning about each other's inner world (worries, hopes, the details of each other's day).
2. **Nurture fondness and admiration:** notice and say what you appreciate.
3. **Turn toward each other:** respond to small bids for attention ("look at this", a sigh, a question) rather than ignoring them.
4. **Let your partner influence you:** be open to their view and share decisions.
5. **Solve your solvable problems:** using gentle start-ups, repair attempts, and compromise.
6. **Overcome gridlock:** on deep, recurring disagreements, aim for dialogue and understanding the dream behind each position, not a final solution.
7. **Create shared meaning:** rituals, roles, goals and values you build together.

A central idea is that many conflicts are **perpetual**: they come from lasting differences and won't be "solved". The goal there is to keep talking about them kindly. (The book gives a figure for how many; the app deliberately doesn't quote one.)

**How the companion uses it:** it suggests small, concrete habits (a daily check-in, noticing bids, saying one appreciation), and helps couples tell solvable problems apart from perpetual ones.

---

## 5. Gottman: the four horsemen and their antidotes

*Key: `reference:gottman-four-horsemen`. Source: John Gottman and Nan Silver, The Seven Principles for Making Marriage Work (1999).*

Four patterns of communication that Gottman found especially harmful, each with a commonly taught antidote:

| Horseman | What it looks like | Antidote |
|---|---|---|
| **Criticism** | Attacking the person ("you always…", "you never…") rather than the behaviour | **Gentle start-up:** "I feel… about… I need…" |
| **Contempt** | Mockery, sarcasm, eye-rolling, a sense of superiority. The most damaging of the four | **Build a culture of appreciation:** regularly notice and say the good |
| **Defensiveness** | Counter-attacking or playing the victim to ward off blame | **Take responsibility,** even for a small part |
| **Stonewalling** | Shutting down and tuning out, often when overwhelmed ("flooded") | **Self-soothing:** take a break (⚑ the book suggests about 20 minutes), then come back |

**How the companion uses it:** when one shows up, it names it gently and without blame, and offers what the antidote might sound like. The debrief asks the couple to spot them *together*. The companion never labels either partner as "a stonewaller" or "contemptuous".

---

## 6. Gottman: gentle start-up and repair attempts

*Key: `reference:gottman-repair-and-startup`. Source: John Gottman and Nan Silver, The Seven Principles for Making Marriage Work (1999).*

- **Start-up:** how a hard conversation begins tends to set how it ends. A harsh start-up (blame, sarcasm) usually goes badly, while a gentle one (describing your own feelings and a specific need, without attacking) gives it a better chance.
- **Repair attempts:** anything that de-escalates during a conflict, such as humour, "let's start over", "I see your point", or a touch. What matters most is whether the other partner is able to *accept* a repair attempt, and that's easier when the relationship has a store of goodwill.
- **Flooding:** when one partner is physically overwhelmed, a productive conversation isn't possible. Taking a break and agreeing when to come back is a skill, not a defeat.

**How the companion uses it:** in the conflict-debrief reflection and in everyday conversation. It might suggest what a gentle start-up could sound like, notice a repair attempt and point it out kindly, or suggest an agreed pause.

---

*The app's copy is in `REFERENCE_DOCS` in `app/memory/memory_store.py`, under the keys above. Seeding skips keys that already exist, so a correction needs a version bump or a re-seed. The README credits you for choosing the books.*
