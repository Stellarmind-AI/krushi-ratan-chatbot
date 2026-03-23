# F1 Confirmation Layer — Frontend Integration Guide

## Overview

When the backend detects that a user query is ambiguous, it **pauses the pipeline** and sends a `clarification_request` message instead of a normal answer. The frontend must render this as a set of clickable option buttons, wait for the user to tap one, then send a `clarification_response` message back. The backend then resumes and returns the normal `text_output` answer.

```
Frontend                              Backend
   │                                     │
   │── text_input: "wheat" ─────────────►│
   │                                     │  F1 detects ambiguity
   │◄── clarification_request ───────────│  (pipeline paused)
   │                                     │
   │  [render buttons to user]           │
   │                                     │
   │── clarification_response ──────────►│
   │   { intent_key: "crop_price" }      │  pipeline resumes
   │                                     │
   │◄── text_output ─────────────────────│  normal answer
```

---

## New Message Types

### Received from Backend — `clarification_request`

```json
{
  "type": "clarification_request",
  "scenario": "crop_name",
  "question": "What information are you looking for?",
  "options": [
    { "label": "📈 Market price at mandi / yard", "intent_key": "crop_price" },
    { "label": "🛒 Buy from K-Shop (farm supplies)",  "intent_key": "kshop_product" },
    { "label": "📦 Sell / buy on marketplace",        "intent_key": "buy_sell_product" },
    { "label": "🌱 Seed / variety information",       "intent_key": "seed_info" }
  ],
  "timestamp": "2025-01-01T10:00:00.000000"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Always `"clarification_request"` |
| `scenario` | string | Internal tag — use for logging only |
| `question` | string | Display this above the buttons |
| `options` | array | Each option has `label` (button text) and `intent_key` (opaque ID) |

### Sent to Backend — `clarification_response`

```json
{
  "type":       "clarification_response",
  "session_id": "abc123",
  "intent_key": "crop_price"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | ✅ | Must be `"clarification_response"` |
| `session_id` | string | ⚠️ Recommended | Same session_id used in the original text_input |
| `intent_key` | string | ✅ | Copied exactly from the chosen option |

---

## The 5 Scenarios

| Scenario | Trigger Example | Options |
|----------|----------------|---------|
| `crop_name` | "wheat", "kapas", "bajra" | Market price / K-Shop / Buy-Sell / Seeds |
| `generic_product` | "product batao", "item joiye", "buy karu" | K-Shop / Buy-Sell / Mandi price / Seeds |
| `price_query` | "rate shu che", "ketla ma mile" | Crop price / K-Shop price / Buy-Sell price |
| `equipment_query` | "tractor", "pump", "machine", "sprayer" | K-Shop new / Buy-Sell used |
| `location_query` | "surat ma", "rajkot thi" | Crop price / Local news / Buy-Sell |

**Note:** If the query already contains an unambiguous source signal (`kshop`, `bhav`, `mandi`, `video`, `news`, `seed`, etc.), the layer does NOT trigger — the pipeline proceeds immediately as before.

---

## Implementation Reference (React / TypeScript)

```tsx
type ClarificationOption = {
  label: string;
  intent_key: string;
};

type ClarificationRequest = {
  type: "clarification_request";
  scenario: string;
  question: string;
  options: ClarificationOption[];
  timestamp: string;
};

// In your WebSocket message handler:
ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);

  if (msg.type === "clarification_request") {
    // Render buttons in chat UI
    setClarification(msg as ClarificationRequest);
    return;
  }

  if (msg.type === "text_output") {
    // Normal answer — clear any pending clarification
    setClarification(null);
    setMessages(prev => [...prev, { role: "assistant", text: msg.text }]);
    return;
  }
};

// When user taps a button:
const handleOptionTap = (intentKey: string) => {
  ws.send(JSON.stringify({
    type:       "clarification_response",
    session_id: currentSessionId,
    intent_key: intentKey,
  }));
  setClarification(null); // remove buttons
};
```

### Button component example

```tsx
{clarification && (
  <div className="clarification-card">
    <p className="question">{clarification.question}</p>
    <div className="options">
      {clarification.options.map((opt) => (
        <button
          key={opt.intent_key}
          onClick={() => handleOptionTap(opt.intent_key)}
          className="option-button"
        >
          {opt.label}
        </button>
      ))}
    </div>
  </div>
)}
```

---

## Flutter / Dart Reference

```dart
void _handleWebSocketMessage(dynamic raw) {
  final msg = jsonDecode(raw as String) as Map<String, dynamic>;

  if (msg['type'] == 'clarification_request') {
    setState(() {
      _pendingClarification = msg;
    });
    return;
  }

  if (msg['type'] == 'text_output') {
    setState(() {
      _pendingClarification = null;
      _messages.add({'role': 'assistant', 'text': msg['text']});
    });
    return;
  }
}

void _sendClarificationResponse(String intentKey) {
  final payload = jsonEncode({
    'type':       'clarification_response',
    'session_id': _sessionId,
    'intent_key': intentKey,
  });
  _channel.sink.add(payload);
  setState(() => _pendingClarification = null);
}

// Widget:
if (_pendingClarification != null) ...[
  Text(_pendingClarification!['question'] as String),
  ...(_pendingClarification!['options'] as List).map((opt) {
    final o = opt as Map<String, dynamic>;
    return ElevatedButton(
      onPressed: () => _sendClarificationResponse(o['intent_key'] as String),
      child: Text(o['label'] as String),
    );
  }),
]
```

---

## Edge Cases

### Session expired / backend restarted
If there is no pending clarification stored server-side (backend restarted, session expired), the backend returns:
```json
{ "type": "error", "error": "No pending clarification for this session. Please send your question again." }
```
**Handle this by**: clearing your clarification UI state and prompting the user to re-send their message.

### Network interruption mid-clarification
Store `_pendingClarification` in local state only (not persisted). On WebSocket reconnect, clear it and the user will naturally retype their query.

### User sends a new text_input while clarification is pending
The new `text_input` starts a fresh pipeline evaluation. The previous pending clarification is overwritten server-side. Clear your clarification UI when you send a new `text_input`.

---

## Intent Keys Reference

| `intent_key` | Meaning | Tables resolved |
|---|---|---|
| `crop_price` | Mandi/yard crop market prices | products, yards, cities, sub_categories, weights |
| `kshop_product` | K-Shop farm equipment & supplies | kshop_products, kshop_companies, kshop_categories, kshop_weights |
| `buy_sell_product` | Buy/Sell marketplace listings | buy_sell_products, buy_sell_categories, users |
| `seed_info` | Crop seed varieties | seeds, sub_categories |
| `local_news` | Agricultural news by area | news, cities, states |
| `video_search` | Farming videos | video_posts, users, video_categories |

---

## Files Changed (Backend Summary)

| File | Change type |
|------|-------------|
| `app/services/agent/confirmation_layer.py` | **New file** — entire F1 logic |
| `app/services/agent/__init__.py` | Added export for new module |
| `app/services/agent/orchestrator.py` | Added `confirmed_intent` param to `process_query` and `_flow_sql` |
| `app/websocket/chat_handler.py` | Added F1 check in `_run_pipeline`, new `_handle_clarification_response` method |

All other files are **unchanged**.
