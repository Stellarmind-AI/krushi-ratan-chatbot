# Expectations vs Catalog — Validation Report

Catalog source: `entity_catalog_v2.json`  |  Expectations: 38 triplets (questions 1–114)

| Status | Meaning |
|---|---|
| ✅ EXACT | entity exists in DB (catalog hit) — data CAN exist |
| 🟡 PARTIAL | substring match — LIKE/IN search will find these canonicals |
| ❌ UNKNOWN | not in catalog — likely AI-invented question keyword OR missing synonym |
| ➖ SKIP | free-text (topic/identifier) — not catalog-checkable |

## Per-question results

| Q# | Intent | Entity (type) | Status | Match |
|---|---|---|---|---|
| 1–3 | greeting | — | — | no entities expected |
| 4–6 | navigation | — | — | no entities expected |
| 7–9 | greeting | — | — | no entities expected |
| 10–12 | general | — | — | no entities expected |
| 13–15 | general | — | — | no entities expected |
| 16–18 | general | — | — | no entities expected |
| 19–21 | general | — | — | no entities expected |
| 22–24 | general | — | — | no entities expected |
| 25–27 | navigation | — | — | no entities expected |
| 28–30 | navigation | — | — | no entities expected |
| 31–33 | navigation | — | — | no entities expected |
| 34–36 | navigation | — | — | no entities expected |
| 37–39 | navigation | — | — | no entities expected |
| 40–42 | navigation | — | — | no entities expected |
| 43–45 | buy_sell_product | ભેંસ (animal) | ✅ EXACT | buy_sell_categories → ભેંસ |
| 43–45 | buy_sell_product | ગાય (animal) | ✅ EXACT | buy_sell_categories → ગાય |
| 46–48 | crop_price | શીંગ મગડી (crop) | ✅ EXACT | sub_categories → શીંગ મગડી |
| 46–48 | crop_price | ભાવનગર (location) | ✅ EXACT | cities → ભાવનગર |
| 49–51 | crop_price | કપાસ (crop) | ✅ EXACT | sub_categories → કપાસ |
| 49–51 | crop_price | તલ કાળા (crop) | ✅ EXACT | sub_categories → તલ કાળા |
| 49–51 | crop_price | ગારીયાધાર (location) | ✅ EXACT | talukas → ગારીયાધાર |
| 52–54 | crop_price | શીંગ કાદરી (crop) | ✅ EXACT | sub_categories → શીંગ કાદરી |
| 52–54 | crop_price | મહુવા (location) | ✅ EXACT | talukas → મહુવા |
| 55–57 | crop_price | બાજરો (crop) | ✅ EXACT | sub_categories → બાજરો |
| 55–57 | crop_price | અનાજ (category) | ✅ EXACT | categories → અનાજ |
| 55–57 | crop_price | કઠોળ (category) | ✅ EXACT | categories → કઠોળ |
| 55–57 | crop_price | કેશોદ (location) | ✅ EXACT | talukas → કેશોદ |
| 58–60 | crop_price | રોકડિયા પાક (category) | ✅ EXACT | categories → રોકડિયા પાક |
| 58–60 | crop_price | તેલીબીયા (category) | ✅ EXACT | categories → તેલીબીયા |
| 58–60 | crop_price | વિસાવદર (location) | ✅ EXACT | talukas → વિસાવદર |
| 61–63 | crop_price | તલ સફેદ (crop) | ✅ EXACT | sub_categories → તલ સફેદ |
| 61–63 | crop_price | સોયાબીન (crop) | ✅ EXACT | sub_categories → સોયાબીન |
| 61–63 | crop_price | જામનગર (location) | ✅ EXACT | cities → જામનગર |
| 64–66 | crop_price | કપાસ (crop) | ✅ EXACT | sub_categories → કપાસ |
| 64–66 | crop_price | રોકડિયા પાક (category) | ✅ EXACT | categories → રોકડિયા પાક |
| 64–66 | crop_price | ગુજરાત (location) | ✅ EXACT | states → ગુજરાત |
| 67–69 | crop_price | શીંગ ગીરનાર (crop) | ✅ EXACT | sub_categories → શીંગ ગીરનાર |
| 67–69 | crop_price | આદું (crop) | ✅ EXACT | sub_categories → આદું |
| 67–69 | crop_price | ગારીયાધાર (location) | ✅ EXACT | talukas → ગારીયાધાર |
| 70–72 | equipment_kshop | સીડર (equipment) | ✅ EXACT | kshop_categories → સીડર |
| 73–75 | kshop_product | મોટર સ્ટાર્ટર (category) | ✅ EXACT | kshop_categories → મોટર સ્ટાર્ટર |
| 73–75 | kshop_product | બેટરી સ્પ્રેયર (category) | ✅ EXACT | kshop_categories → બેટરી સ્પ્રેયર |
| 76–78 | equipment_kshop | સબમર્સિબલ પંપ (equipment) | ✅ EXACT | kshop_categories → સબમર્સિબલ પંપ |
| 79–81 | kshop_product | બ્રશ કટર (category) | ✅ EXACT | kshop_categories → બ્રશ કટર |
| 79–81 | kshop_product | અર્થ ઓગર્સ (category) | ✅ EXACT | kshop_categories → અર્થ ઓગર્સ |
| 82–84 | equipment_kshop | ઝટકા મશીન (equipment) | ✅ EXACT | kshop_categories → ઝટકા મશીન |
| 85–87 | equipment_used | ટ્રેક્ટર (equipment) | ✅ EXACT | buy_sell_categories → ટ્રેક્ટર |
| 85–87 | equipment_used | ટ્રોલી (equipment) | ✅ EXACT | buy_sell_categories → ટ્રોલી |
| 88–90 | buy_sell_product | ગાય (animal) | ✅ EXACT | buy_sell_categories → ગાય |
| 88–90 | buy_sell_product | ગીર (topic) | ➖ SKIP | free text |
| 91–93 | equipment_used | થ્રેશર (equipment) | ✅ EXACT | kshop_categories → થ્રેશર |
| 94–96 | buy_sell_product | ઘેટા બકરા (animal) | ✅ EXACT | buy_sell_categories → ઘેટા બકરા |
| 94–96 | buy_sell_product | ઘોડો (animal) | ✅ EXACT | buy_sell_categories → ઘોડો |
| 97–99 | buy_sell_product | 1778741216208 (identifier) | ➖ SKIP | free text |
| 100–102 | news | સરકાર અને યોજના સમાચાર (news_type) | ✅ EXACT | news_types → સરકાર અને યોજના સમાચાર |
| 100–102 | news | ગુજરાત (location) | ✅ EXACT | states → ગુજરાત |
| 100–102 | news | સબસિડી (topic) | ➖ SKIP | free text |
| 103–105 | news | હવામાન અને ચેતવણી સમાચાર (news_type) | ✅ EXACT | news_types → હવામાન અને ચેતવણી સમાચાર |
| 103–105 | news | ભાવનગર (location) | ✅ EXACT | cities → ભાવનગર |
| 103–105 | news | ગારીયાધાર (location) | ✅ EXACT | talukas → ગારીયાધાર |
| 103–105 | news | અતિવૃષ્ટિ (topic) | ➖ SKIP | free text |
| 103–105 | news | વાવાઝોડું (topic) | ➖ SKIP | free text |
| 106–108 | news | જૂનાગઢ (location) | ✅ EXACT | cities → જૂનાગઢ |
| 106–108 | news | વિસાવદર (location) | ✅ EXACT | talukas → વિસાવદર |
| 106–108 | news | પાક વીમા યોજના (topic) | ➖ SKIP | free text |
| 109–111 | news | હવામાન અને ચેતવણી સમાચાર (news_type) | ✅ EXACT | news_types → હવામાન અને ચેતવણી સમાચાર |
| 109–111 | news | કેશોદ (location) | ✅ EXACT | talukas → કેશોદ |
| 109–111 | news | વરસાદ આગાહી (topic) | ➖ SKIP | free text |
| 112–114 | news | મુખ્યમંત્રી કિસાન સહાય યોજના (topic) | ➖ SKIP | free text |

## Summary

- ✅ EXACT: 47   🟡 PARTIAL: 0   ❌ UNKNOWN: 0   ➖ SKIP: 8

## Review notes (label decisions awaiting confirmation)

- Q4–6 (navigation): Greeting + real question hybrid — the question part ('how to start chatting') wins per routing rule. Alternative acceptable label: general.
- Q7–9 (greeting): Conversational presence-check — a warm greeting-style reply is correct. Alternative: general.
- Q13–15 (general): Should match the 'is the app free' entry in general_questions.json
- Q16–18 (general): Needs a contact/support entry in general_questions.json — verify one exists with the real number
- Q19–21 (general): Verify general_questions.json has a company/ownership entry; if not, honest 'no info' is the correct answer
- Q25–27 (navigation): Explicit SELL process → navigation (Buy/Sell listing steps). Entities (ટ્રેક્ટર, સાંઢ) are context, not search keywords.
- Q28–30 (navigation): DECIDED (owner, 2026-06-12): 'where can I watch X / open that page' → NAVIGATION — guide the user to the videos screen, no DB search. Explicit list requests ('show me videos about X') remain video/SQL.
- Q31–33 (navigation): DECIDED (owner, 2026-06-12): same policy as Q28-30 — 'where does this video appear' → NAVIGATION, guide to videos screen.
- Q34–36 (navigation): Explicit BUY process ('which section do I visit to place the order') → navigation per agreed rule. If user had asked its price → equipment_kshop/SQL.
- Q37–39 (navigation): App may have no subsidy-form feature — honest 'feature not available, schemes appear in news section' is a correct answer. Acceptable alternative: news.
- Q40–42 (navigation): User asks WHERE the page is → navigation. If they asked for the forecast itself → news/SQL. Borderline — confirm.
- Q43–45 (buy_sell_product): DECIDED (owner, 2026-06-12): SQL confirmed — the bot fetches the listings from the database.
- Q46–48 (crop_price): Hindi variant says મૂંગફળી મગડી — must resolve to the same canonical શીંગ મગડી.
- Q49–51 (crop_price): MULTI-ITEM: two crops must be searched separately, never concatenated. User said 'taluka' explicitly → taluka-level filter is correct.
- Q52–54 (crop_price): VARIETY CASE: શીંગ + કાદરી must resolve together. Validator must confirm શીંગ કાદરી exists in DB — this question may be Claude-invented.
- Q55–57 (crop_price): Canonical could be બાજરો or બાજરી — validator will tell. Category words are secondary; direct crop match should win.
- Q58–60 (crop_price): CATEGORY-SCOPED LIST: no crop keyword — filter via categories, list crops with prices. Two categories = OR, not concatenated.
- Q61–63 (crop_price): MULTI-ITEM + aggregation: lowest price per yard.
- Q64–66 (crop_price): ગુજરાત is a STATE — must land on states filter, never city/taluka/yard.
- Q67–69 (crop_price): VARIETY (શીંગ ગીરનાર) + plain crop (આદું) in one query — both must survive extraction.
- Q70–72 (equipment_kshop): 'new' + 'company' → K-Shop, no clarification needed (condition explicit).
- Q73–75 (kshop_product): Output entity is COMPANIES, filter entity is two kshop categories — must not merge the two categories into one string.
- Q76–78 (equipment_kshop): kshop_products has discount_price column — answer should compare price vs discount_price.
- Q82–84 (equipment_kshop): Warranty info likely not in DB — answer gives price, honestly omits warranty. Buy-verb present BUT price asked → SQL not navigation (the agreed nuance).
- Q85–87 (equipment_used): MULTI-ITEM, 'second hand' → buy_sell, no clarification (condition explicit).
- Q88–90 (buy_sell_product): ગીર is a breed qualifier inside free-text listing names — category filter = ગાય, breed = secondary LIKE on product rows. Confirm handling.
- Q91–93 (equipment_used): 'Nearby' is UNRESOLVABLE (no user location available) — correct behavior: search without location filter, do NOT invent one. Hallucination check.
- Q94–96 (buy_sell_product): Two categories, ORDER BY created_at DESC LIMIT — date is the answer.
- Q97–99 (buy_sell_product): DECIDED (owner, 2026-06-12): never expose seller phone numbers. The pipeline may look up the listing, but the answer redirects to the in-app contact guide. Enforce via privacy sanitization + answer prompt; verify in baseline and Phase 3.
- Q100–102 (news): DB canonical confirmed: 'સરકાર અને યોજના સમાચાર'. User phrase 'સરકારી યોજના' maps via entity_catalog_manual.json.
- Q103–105 (news): MULTI-LOCATION (city + taluka). 'Today' explicitly said → date filter is correct here.
- Q109–111 (news): User explicitly said 'this month' → date filter correct.
- Q112–114 (news): Answer = the news_type of matching news rows (reverse lookup: topic → which type).
