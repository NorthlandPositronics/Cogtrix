# Using Cogtrix as a Telegram Assistant

This guide walks you through turning Cogtrix into a personal Telegram assistant. Once set up, you can send and receive messages, photos, and manage contacts -- all through natural language commands in Cogtrix's command line.

## Table of Contents

- [What Can Cogtrix Do with Telegram?](#what-can-cogtrix-do-with-telegram)
- [How It Works](#how-it-works)
- [Step-by-Step Setup](#step-by-step-setup)
- [Finding Your Chat ID](#finding-your-chat-id)
- [Configuration Deep Dive](#configuration-deep-dive)
- [Usage Examples](#usage-examples)
- [Real-World Scenarios](#real-world-scenarios)
- [Using Docker Compose](#using-docker-compose)
- [Security and Privacy](#security-and-privacy)
- [Troubleshooting](#troubleshooting)

---

## What Can Cogtrix Do with Telegram?

Once set up, you can ask Cogtrix things like:

```
You: Send a Telegram message to alice saying "Deployment finished successfully"
You: Check my Telegram messages
You: Send the team channel a summary of today's top tech news
You: Show my Telegram contacts
You: Send alice a photo from https://example.com/chart.png with caption "Q4 results"
```

Cogtrix picks the right tool automatically. It can also chain Telegram with other tools -- for example, searching the web and sending the results to a Telegram chat.

**Available tools:**

| Tool | What it does |
|------|-------------|
| `telegram_send` | Send a text message to a chat ID, @username, or phonebook contact |
| `telegram_send_photo` | Send a photo (by URL) with an optional caption |
| `telegram_check` | Retrieve recent messages sent to the bot |
| `telegram_contacts` | List your configured phonebook and active filter rules |

---

## How It Works

Cogtrix connects to Telegram through the official **[Telegram Bot API](https://core.telegram.org/bots/api)**. You create a bot with [@BotFather](https://t.me/BotFather), give Cogtrix the bot token, and it can send messages on behalf of the bot and read messages that users send to it.

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Telegram    │◄───►│  Telegram Bot    │◄───►│     Cogtrix      │
│  Users       │     │  API (cloud)     │     │  (AI Assistant)  │
└─────────────┘     └──────────────────┘     └──────────────────┘
  Chat with bot       HTTPS requests          Natural language
```

**Key difference from WhatsApp:** Telegram bots are first-class citizens in the Telegram ecosystem. There's no QR code scanning, no Docker container to run, and no session to maintain. You just need a bot token.

**Important things to know about Telegram bots:**

- Bots can only receive messages from users who have **started a conversation** with the bot first (sent `/start`).
- Bots can send messages to any user or group that has previously interacted with them.
- Bots can be added to groups and channels as members.
- Messages are sent as the bot (with the bot's name and avatar), not as you personally.

---

## Step-by-Step Setup

### Step 1: Create a Telegram bot

1. Open Telegram and search for **@BotFather** (or open [t.me/BotFather](https://t.me/BotFather)).
2. Send `/newbot`.
3. BotFather will ask for a **display name** -- enter something like `Cogtrix Assistant`.
4. BotFather will ask for a **username** -- enter something like `my_cogtrix_bot` (must end in `bot`).
5. BotFather replies with your **bot token**. It looks like this:

```
7123456789:AAHb2Txq-Wm3kZxYdL0P9aF5vKJcR0zE4Mk
```

> **Keep this token secret.** Anyone with this token can control your bot.

### Step 2: Start a chat with your bot

Before Cogtrix can send you messages, you need to open a conversation with the bot:

1. In Telegram, search for your bot by username (e.g. `@my_cogtrix_bot`).
2. Press **Start** or send `/start`.
3. The bot won't reply yet -- that's normal. Cogtrix will handle responses.

If you want the bot to message a **group**, add the bot to the group and send any message there.

### Step 3: Configure Cogtrix

**Option A: Environment variable (simplest)**

```bash
export COGTRIX_TELEGRAM_TOKEN="7123456789:AAHb2Txq-Wm3kZxYdL0P9aF5vKJcR0zE4Mk"
python cogtrix.py
```

That's it. The Telegram tools appear automatically.

**Option B: Config file (recommended for phonebook)**

Create or edit `.cogtrix.yaml`:

```yaml
services:
  telegram:
    bot_token: "7123456789:AAHb2Txq-Wm3kZxYdL0P9aF5vKJcR0zE4Mk"
    phonebook:
      alice: "123456789"
      bob: "987654321"
      team: "-1001234567890"
```

The phonebook lets you say `"send alice a message"` instead of `"send 123456789 a message"`. Group chat IDs start with `-100`.

### Step 4: Verify

```bash
python cogtrix.py
```

```
You: /tools telegram
```

You should see `telegram_send`, `telegram_send_photo`, `telegram_check`, and `telegram_contacts` listed.

Now try it:

```
You: Check my Telegram messages
You: Send alice "Hello from Cogtrix!"
```

You'll see a confirmation prompt before the message is sent:

```
Tool: telegram_send
  to: alice
  message: Hello from Cogtrix!
Allow? [y/n/all]:
```

Type `y` to approve. The message appears in Telegram within seconds.

---

## Finding Your Chat ID

Telegram identifies every user, group, and channel by a numeric **chat ID**. You need this ID to add contacts to your phonebook or filter list.

### Method 1: Use Cogtrix itself

After starting a chat with your bot, run:

```
You: Check my Telegram messages
```

The output includes the chat ID for each sender:

```
Recent Telegram messages (1):

  [2026-02-14 10:30] alice in Alice: /start
```

The chat ID is the numeric value shown in the raw message. To see it, ask:

```
You: Check Telegram messages and show the chat IDs
```

### Method 2: Use @userinfobot

1. Open Telegram and search for `@userinfobot`.
2. Send `/start` -- it replies with your numeric user ID.

### Method 3: For groups

1. Add your bot to the group.
2. Send a message in the group.
3. Run `telegram_check` in Cogtrix -- the group's chat ID appears (starts with `-100`).

---

## Configuration Deep Dive

### All Telegram options

```yaml
services:
  telegram:
    bot_token: "7123456789:AAHb..."
    allow_send: true
    allow_receive: true
    require_confirmation: true
    filter_mode: allow
    contacts: ["123456789", "987654321"]
    phonebook:
      alice: "123456789"
      bob: "987654321"
      team: "-1001234567890"
      alerts: "-1009876543210"
    rate_limit: 30
    max_message_length: 4096
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `bot_token` | string | -- | Bot token from @BotFather (required) |
| `allow_send` | bool | `true` | Enable send tools |
| `allow_receive` | bool | `true` | Enable receive/check tools |
| `require_confirmation` | bool | `true` | Ask for user approval before sending |
| `filter_mode` | string | `"none"` | `"none"`, `"allow"`, `"ignore"`, or `"blacklist"` (legacy `"whitelist"` is accepted and auto-mapped to `"allow"`) |
| `contacts` | array | `[]` | Chat IDs for the filter list |
| `phonebook` | object | `{}` | Nickname-to-chat-ID mapping |
| `rate_limit` | int | `30` | Max outbound messages per hour (0 = unlimited) |
| `max_message_length` | int | `4096` | Truncate outgoing messages beyond this length |
| `ignore_older_than` | string | -- | Skip messages older than this duration (e.g. `"24h"`, `"7d"`) |

### Environment variables

All options can be set via environment variables (useful in Docker or CI):

| Variable | Description |
|----------|-------------|
| `COGTRIX_TELEGRAM_TOKEN` | Bot token from @BotFather |
| `COGTRIX_TELEGRAM_SEND` | `true` / `false` |
| `COGTRIX_TELEGRAM_RECEIVE` | `true` / `false` |
| `COGTRIX_TELEGRAM_FILTER` | `none` / `allow` / `ignore` / `blacklist` |
| `COGTRIX_TELEGRAM_CONTACTS` | Comma-separated chat IDs |

### Contact filtering

Control who the bot can communicate with:

| Mode | Behavior |
|------|----------|
| `none` (default) | Respond to all contacts |
| `allow` | Only respond to chat IDs in the `contacts` list (legacy value `whitelist` is accepted and auto-mapped to `allow`) |
| `ignore` | Skip messages from chat IDs in the `contacts` list silently |
| `blacklist` | Delete the message for chat IDs in the `contacts` list (Telegram has no archive API, so the chat is not archived) |

Phonebook nicknames are resolved automatically. If `alice` maps to `123456789` and `123456789` is in the allow list, then `"send alice a message"` works.

### Rate limiting

The default rate limit is 30 messages per hour. This prevents accidental message floods. Set to `0` for unlimited. The limit uses an in-memory sliding window that resets when Cogtrix restarts.

### Read-only mode

To let the agent read Telegram messages but never send:

```yaml
services:
  telegram:
    bot_token: "7123456789:AAHb..."
    allow_send: false
    allow_receive: true
```

---

## Usage Examples

### Basic messaging

```
You: Send a Telegram message to alice saying "Build passed, ready for review"
You: Send 123456789 "Meeting in 10 minutes"
You: Check my Telegram messages
You: Show my Telegram contacts and filter settings
```

### Sending to groups

```
You: Send the team channel "Deployment to production completed at 14:30 UTC"
You: Send alerts "WARNING: Disk usage on server-03 is at 92%"
```

(This uses phonebook entries like `"team": "-1001234567890"` and `"alerts": "-1009876543210"`.)

### Sending photos

```
You: Send alice a photo from https://example.com/chart.png with caption "Q4 results"
```

### Combining with other tools

```
You: Search the web for the latest Python 3.14 features and send a summary to alice via Telegram
You: Read the file report.md and send the key points to the team channel on Telegram
You: Check my Telegram messages and summarize what alice said
You: What's the weather in London? Then send the forecast to bob on Telegram
```

---

## Real-World Scenarios

### DevOps notifications

Set up Cogtrix as part of your workflow to send deployment notifications:

```bash
python cogtrix.py --prompt "Send the team channel on Telegram: Deployment v2.4.1 to production completed successfully at $(date -u +%H:%M) UTC"
```

### Morning briefing

```
You: Search for top 5 tech news today, summarize each in one sentence, and send the briefing to alice on Telegram
```

Cogtrix will:
1. Use `web_search` to find articles
2. Summarize them
3. Use `telegram_send` to deliver the briefing

### Monitoring assistant

```
You: Check my Telegram messages from the alerts channel and summarize any issues reported in the last hour
```

### Research pipeline

```
You: /think Compare Redis vs Memcached for session storage with 10K concurrent users

(after deep reasoning completes)

You: Send the analysis to bob on Telegram
```

---

## Using Docker Compose

A production-ready `docker-compose.yml` is included in the repository at
[`docker/docker-compose.yml`](../docker/docker-compose.yml). To enable Telegram,
add your bot token to `docker/config/cogtrix.env`:

```bash
COGTRIX_TELEGRAM_TOKEN=7123456789:AAHb2Txq-...
```

Then start the stack:

```bash
cp docker/config/cogtrix.env.example docker/config/cogtrix.env   # add your token
cp docker/config/cogtrix.yml.example docker/config/cogtrix.yml   # adjust settings
cd docker && docker compose up -d
```

You can combine Telegram and WhatsApp in the same deployment -- both sets of tools will be available simultaneously.

---

## Security and Privacy

1. **Confirmation prompts**: By default, Cogtrix asks for your approval before sending any message. You see the recipient and message text, and can approve (`y`), deny (`n`), or approve all Telegram sends for the session (`all`).

2. **Contact filtering**: Use `filter_mode: allow` to restrict the bot to a known set of chat IDs. This prevents the agent from messaging arbitrary users or groups.

3. **Rate limiting**: The hourly rate limit (default 30) prevents accidental message floods.

4. **Message truncation**: Messages longer than `max_message_length` (default 4096) are automatically truncated.

5. **Token security**: The bot token grants full control over the bot. Store it in an environment variable or config file, never commit it to version control. The `.gitignore` already excludes `.cogtrix.json` and `.env*` files.

6. **Bot isolation**: The bot is a separate Telegram account. It cannot read your personal messages or access your contacts. It only sees messages that users send directly to it (or in groups where it's a member).

7. **Revocation**: You can revoke the bot token at any time by messaging @BotFather and using `/revoke`. This immediately disables all access.

---

## Troubleshooting

### Telegram tools not appearing in `/tools`

**Cause:** Bot token is not configured, or both send/receive are disabled.

**Fix:**
1. Check that `COGTRIX_TELEGRAM_TOKEN` is set (or `services.telegram.bot_token` in config)
2. Verify the token is valid: `curl https://api.telegram.org/bot<YOUR_TOKEN>/getMe`
3. Make sure `allow_send` and `allow_receive` are not both `false`

### "Telegram bot token not configured"

**Cause:** The token is missing from both environment and config file.

**Fix:**
```bash
export COGTRIX_TELEGRAM_TOKEN="7123456789:AAHb2Txq-..."
python cogtrix.py
```

### "Bad Request: chat not found"

**Cause:** The recipient hasn't started a conversation with the bot, or the chat ID is wrong.

**Fix:**
1. The recipient must open the bot in Telegram and press **Start** (or send `/start`)
2. For groups: add the bot as a member first
3. Verify the chat ID is correct (see [Finding Your Chat ID](#finding-your-chat-id))

### Bot sends messages but receives none

**Cause:** No one has messaged the bot, or the messages were already consumed by a previous `getUpdates` call.

**Fix:**
- Telegram's `getUpdates` returns each message only once. If you called `telegram_check` and the messages were shown, they won't appear again in the next call.
- Send a new test message to the bot from Telegram, then run `telegram_check`.

### "Blocked: Contact not in allow list"

**Cause:** Contact filtering is enabled (`filter_mode: allow`) and the chat ID isn't in the `contacts` list.

**Fix:** Add the chat ID to `contacts` in your config, or change `filter_mode` to `"none"`.

### "Rate limit reached"

**Cause:** You've sent too many messages in the last hour.

**Fix:** Wait for the sliding window to clear (messages older than 1 hour drop off), or set `rate_limit` to `0` for unlimited.

### Messages are truncated

**Cause:** The message exceeds `max_message_length` (default 4096 characters).

**Fix:** Increase `max_message_length` in config, or ask the agent to summarize/shorten the content before sending.

---

## Telegram vs WhatsApp: Which Should I Use?

| Aspect | Telegram | WhatsApp |
|--------|----------|----------|
| **Setup** | Token from @BotFather (2 min) | Docker container + QR scan (5 min) |
| **Infrastructure** | None -- cloud API | Requires Waha container |
| **Identity** | Bot account (separate from you) | Your personal account |
| **Groups** | Bots are native members | Via WhatsApp Web bridge |
| **Initiation** | Users must `/start` the bot first | Send to any contact |
| **Privacy** | Bot sees only its own messages | Bridge sees all chats |
| **Cost** | Free forever | Free (Waha open source) |

You can use **both simultaneously** -- configure both `services.whatsapp` and `services.telegram` in your config, and Cogtrix will have all eight messaging tools available.

---

## See Also

- [Configuration Reference -- Telegram](CONFIGURATION.md#telegram-messaging) -- Full option table
- [Tools Reference -- Telegram](TOOLS_REFERENCE.md#telegram-messaging) -- Tool parameters
- [WhatsApp Guide](WHATSAPP_GUIDE.md) -- Set up WhatsApp alongside Telegram
- [Dockerfile](../docker/Dockerfile) -- Docker image build
