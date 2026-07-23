# Using Cogtrix as a WhatsApp Assistant

This guide walks you through turning Cogtrix into a personal WhatsApp assistant that can send and receive messages, images, and manage your contacts -- all through natural language commands.

## Table of Contents

- [What Can Cogtrix Do with WhatsApp?](#what-can-cogtrix-do-with-whatsapp)
- [How It Works](#how-it-works)
- [Step-by-Step Setup](#step-by-step-setup)
- [Using Docker Compose (Recommended)](#using-docker-compose-recommended)
- [Configuration Deep Dive](#configuration-deep-dive)
- [Usage Examples](#usage-examples)
- [Security and Privacy](#security-and-privacy)
- [Troubleshooting](#troubleshooting)

---

## What Can Cogtrix Do with WhatsApp?

Once set up, you can ask Cogtrix things like:

```
You: Send a WhatsApp message to alice saying "Meeting moved to 3pm"
You: Check my WhatsApp messages
You: Send bob a summary of today's top tech news
You: Check if alice replied to my last message
You: Show my WhatsApp contacts
```

Cogtrix will use its WhatsApp tools (`whatsapp_send`, `whatsapp_check`, `whatsapp_send_image`, `whatsapp_contacts`) automatically based on your request. It can also combine WhatsApp with other tools -- for example, searching the web and then sending a summary to a contact.

**Available tools:**

| Tool | What it does |
|------|-------------|
| `whatsapp_send` | Send a text message to a phone number or phonebook contact |
| `whatsapp_send_image` | Send an image (by URL) with an optional caption |
| `whatsapp_check` | Retrieve recent messages, optionally filtered by contact |
| `whatsapp_contacts` | List your configured phonebook and active filter rules |

---

## How It Works

Cogtrix connects to WhatsApp through **[Waha](https://waha.devlike.pro/)** -- a self-hosted Docker container that wraps WhatsApp Web behind a REST API. Waha runs on your machine (or server), you scan a QR code with your phone once, and from that point Cogtrix can send and receive messages through the Waha API.

```
┌─────────────┐     ┌─────────────┐     ┌──────────────────┐
│  Your Phone  │◄───►│    Waha      │◄───►│     Cogtrix      │
│  (WhatsApp)  │     │  (Docker)   │     │  (AI Assistant)  │
└─────────────┘     └─────────────┘     └──────────────────┘
   QR scan once       REST API            Natural language
```

**Important:** Waha uses your personal WhatsApp account. Messages are sent as you. This is not a business API -- it's a bridge to WhatsApp Web.

---

## Step-by-Step Setup

### Step 1: Start the Waha container

```bash
docker run -d --name waha -p 3000:3000 devlikeapro/waha
```

This starts Waha on `http://localhost:3000`. The `-d` flag runs it in the background.

### Step 2: Link your WhatsApp account

1. Open `http://localhost:3000` in your browser.
2. You'll see a QR code on the dashboard.
3. On your phone, open WhatsApp > **Settings** > **Linked Devices** > **Link a Device**.
4. Scan the QR code.
5. Wait a few seconds -- the status should change to **WORKING**.

> **Tip:** The session persists across container restarts if you mount a volume (see Docker Compose section below).

### Step 3: Configure Cogtrix

The simplest approach -- no config file needed:

```bash
python cogtrix.py
```

Cogtrix auto-detects Waha on `localhost:3000`. If it's reachable, the WhatsApp tools appear automatically.

For a more complete setup, add a `services.whatsapp` section to your `.cogtrix.yaml`:

```yaml
services:
  whatsapp:
    waha_url: "http://localhost:3000"
    phonebook:
      alice: "+14155551234"
      bob: "+442071234567"
      team: "+491701234567"
```

The phonebook lets you say `"send alice a message"` instead of `"send +14155551234 a message"`.

### Step 4: Verify

```bash
python cogtrix.py
```

```
You: /tools whatsapp
```

You should see `whatsapp_send`, `whatsapp_send_image`, `whatsapp_check`, and `whatsapp_contacts` listed.

```
You: Show my WhatsApp contacts
You: Check my recent WhatsApp messages
You: Send alice "Hello from Cogtrix!"
```

---

## Using Docker Compose

> **Note:** A production-ready `docker-compose.yml` is not yet included in the
> repository. The section below describes the intended setup for when it ships.

For a production-like deployment, use Docker Compose to run both Cogtrix and
Waha together. You will need a `docker-compose.yml` that defines:

- **Waha** (`devlikeapro/waha`) on port 3000 (dashboard for QR code scanning)
- **Cogtrix** with `COGTRIX_WHATSAPP_URL=http://waha:3000`
- Volumes for session history and Waha session data

```bash
docker compose up -d
docker compose exec cogtrix python cogtrix.py
```

### First-time setup with Docker Compose

1. `docker compose up -d` -- start both containers
2. Open `http://localhost:3000` -- scan QR code with your phone
3. `docker compose exec cogtrix python cogtrix.py` -- start chatting
4. `You: Check my WhatsApp messages` -- verify it works

---

## Configuration Deep Dive

### All WhatsApp options

```yaml
services:
  whatsapp:
    waha_url: "http://localhost:3000"
    api_key: "yoursecretkey"
    session: default
    allow_send: true
    allow_receive: true
    require_confirmation: true
    filter_mode: whitelist
    contacts: ["+14155551234", "+442071234567"]
    phonebook:
      alice: "+14155551234"
      bob: "+442071234567"
    rate_limit: 30
    max_message_length: 4096
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `waha_url` | string | `http://localhost:3000` | Waha server URL |
| `api_key` | string | -- | Waha `X-Api-Key` header (set this if you secured Waha) |
| `session` | string | `default` | Waha session name |
| `allow_send` | bool | `true` | Enable send tools |
| `allow_receive` | bool | `true` | Enable receive/check tools |
| `require_confirmation` | bool | `true` | Ask for user approval before sending |
| `filter_mode` | string | `none` | `"none"`, `"whitelist"`, or `"blacklist"` |
| `contacts` | array | `[]` | Phone numbers for the filter list (E.164 format) |
| `phonebook` | object | `{}` | Nickname-to-number mapping |
| `rate_limit` | int | `30` | Max outbound messages per hour (0 = unlimited) |
| `max_message_length` | int | `4096` | Truncate outgoing messages beyond this length |

### Environment variables

All options can also be set via environment variables (useful in Docker):

| Variable | Description |
|----------|-------------|
| `COGTRIX_WHATSAPP_URL` | Waha server URL |
| `COGTRIX_WHATSAPP_API_KEY` | Waha API key |
| `COGTRIX_WHATSAPP_SESSION` | Waha session name |
| `COGTRIX_WHATSAPP_SEND` | `true` / `false` |
| `COGTRIX_WHATSAPP_RECEIVE` | `true` / `false` |
| `COGTRIX_WHATSAPP_FILTER` | `none` / `whitelist` / `blacklist` |
| `COGTRIX_WHATSAPP_CONTACTS` | Comma-separated E.164 numbers |

### Contact filtering

Control who the agent can message:

| Mode | Behavior |
|------|----------|
| `none` (default) | All contacts allowed |
| `whitelist` | Only numbers in the `contacts` list can send/receive |
| `blacklist` | Numbers in the `contacts` list are blocked |

Phonebook nicknames are resolved automatically. If `alice` maps to `+14155551234` and `+14155551234` is in the whitelist, then `"send alice a message"` works.

### Rate limiting

The default rate limit is 30 messages per hour. This prevents accidental message floods. Set to `0` for unlimited. The limit uses an in-memory sliding window that resets when Cogtrix restarts.

### Read-only mode

To let the agent read WhatsApp messages but never send:

```yaml
services:
  whatsapp:
    allow_send: false
    allow_receive: true
```

---

## Usage Examples

### Basic messaging

```
You: Send a WhatsApp message to +14155551234 saying "I'll be 10 minutes late"
You: Send alice "Can you review the PR?"
You: Check my WhatsApp messages
You: Check messages from bob
```

### Combining with other tools

```
You: Search the web for today's weather in London, then send alice a summary
You: Read the file report.md and send the key points to bob via WhatsApp
You: Check my WhatsApp messages from alice and summarize what she said
```

### Sending images

```
You: Send alice an image from https://example.com/chart.png with caption "Q4 results"
```

### Checking contact list

```
You: Show my WhatsApp contacts and filter settings
```

---

## Security and Privacy

1. **Confirmation prompts**: By default, Cogtrix asks for your approval before sending any message. You see the recipient and message text, and can approve (`y`), deny (`n`), or approve all WhatsApp sends for the session (`all`).

2. **Contact filtering**: Use whitelist mode to restrict the agent to a known set of contacts. This prevents the agent from messaging arbitrary numbers.

3. **Rate limiting**: The hourly rate limit (default 30) prevents accidental message floods.

4. **Message truncation**: Messages longer than `max_message_length` (default 4096) are automatically truncated.

5. **Self-hosted**: Waha runs on your infrastructure. No third-party services see your messages beyond WhatsApp itself.

6. **No persistent credentials**: Waha uses WhatsApp Web's session mechanism. If you unlink the device from your phone, access is revoked immediately.

---

## Troubleshooting

### WhatsApp tools not appearing in `/tools`

**Cause:** Waha is not reachable or both send/receive are disabled.

**Fix:**
1. Check Waha is running: `curl http://localhost:3000/api/sessions`
2. Check your config: `allow_send` and `allow_receive` should not both be `false`
3. Make sure `requests` is installed: `pip install requests`

### "Cannot connect to Waha server"

**Cause:** Waha container is not running or wrong URL.

**Fix:**
```bash
docker ps | grep waha         # Is it running?
docker logs waha              # Check for errors
curl http://localhost:3000    # Is it reachable?
```

### QR code expired or session disconnected

**Cause:** WhatsApp session needs re-linking.

**Fix:**
1. Open `http://localhost:3000`
2. Re-scan the QR code from your phone
3. Wait for status to show **WORKING**

### "Blocked: Contact not in whitelist"

**Cause:** Contact filtering is enabled and the number isn't in the list.

**Fix:** Add the number to `contacts` in your config, or change `filter_mode` to `"none"`.

### Messages not being received

**Cause:** `allow_receive` may be `false`, or contact filter is blocking inbound messages.

**Fix:** Check config, ensure `allow_receive: true` and the sender is in your whitelist (if using whitelist mode).

---

## See Also

- [Configuration Reference -- WhatsApp](CONFIGURATION.md#whatsapp-messaging) -- Full option table
- [Tools Reference -- WhatsApp](TOOLS_REFERENCE.md#whatsapp-messaging) -- Tool parameters
- [Dockerfile](../Dockerfile) -- Docker image build
