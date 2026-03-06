# Demo Day Quick Guide
## Innovathon 2026 — Krishna River Pollution Intelligence System

---

## Night Before — Checklist
- [ ] Run: python pipeline/data_audit.py → must show GREEN LIGHT
- [ ] Run: streamlit run dashboard/command_center.py → check all 3 tabs
- [ ] Click one dot on map → confirm blue arrow appears
- [ ] Click Tab 3 → confirm Download Report button works
- [ ] Copy entire Main/ folder to USB drive as backup
- [ ] Charge laptop to 100%
- [ ] Save phone hotspot credentials as backup internet

---

## At Venue — Before Your Turn
- [ ] Open terminal in Main/ folder
- [ ] Run: streamlit run dashboard/command_center.py
- [ ] Confirm dashboard opens at http://localhost:8501
- [ ] Pre-zoom map to Krishna River area
- [ ] Pre-click one red dot to confirm arrow works
- [ ] Minimize terminal — keep running in background
- [ ] Put laptop in presentation mode — hide taskbar

---

## Presentation Flow

### Person 1 — Slides 1 to 6 (3-4 minutes)
- Slide 1: Title + SDG connection (15 sec)
- Slide 2: Problem — 4M people, 72 hours, 5 stations (30 sec)
- Slide 3: Solution — 3 layers, 72hrs vs 10mins (30 sec)
- Slide 4: AI Model — dual branch CNN explanation (45 sec)
- Slide 5: Results — 2764 zones, risk distribution (30 sec)
- Slide 6: Impact + Future roadmap (20 sec)
- Hand over to Person 3 for demo

### Person 3 — Live Demo (90 seconds)
1. **Tab 1 → Point to KPI cards (20 sec)**
   Say: "2764 zones, 167 NTU average, 308 critical zones"
2. **Tab 2 → Satellite map (50 sec)**
   Zoom in → Click red dot → Show arrow → Read detail panel
   Say: "Click any zone — see NTU, risk, source, flow direction"
3. **Tab 3 → AI Report (20 sec)**
   Say: "Auto-generated report — downloadable for authorities"
   Click download button

### Q&A — Who Answers What
| Topic | Person |
|-------|--------|
| Problem / Impact / SDG | Person 1 |
| AI model / CNN / Training | Person 2 |
| Dashboard / Map / Tech stack | Person 3 |
| Synthetic data defense | Person 1 |
| Scalability to other rivers | Person 1 |

---

## Emergency Situations

### If dashboard crashes
1. Stay calm — say "Let me show our pre-run analysis"
2. Open sample_dashboard.png, sample_map.png, sample_report.png
3. Walk through images confidently
4. Restart streamlit in background terminal

### If internet cuts out
1. ArcGIS tiles may not load — map shows blank
2. Say: "Satellite tiles require internet — here is our cached analysis"
3. Show the sample images
4. All other tabs still work without internet

### If laptop dies
1. Use USB backup on another laptop
2. Only needs Python + pip install -r requirements.txt
3. Then: streamlit run dashboard/command_center.py

---

## Three Lines to Remember

**Person 1:**
"72 hours of manual testing replaced by 10 minute AI
 detection for 4 million people on Krishna River."

**Person 2:**
"Dual branch CNN fuses satellite spectral data with
 geographic coordinates — giving spatial pollution
 intelligence no single model can match."

**Person 3:**
"Click any zone on the real satellite map — instantly
 see pollution level, health score, source, and exactly
 where it is flowing next."

---

## One Universal Emergency Line

If anyone blanks out completely, say:

"We replaced 72 hours of manual river testing with a
 10 minute AI pipeline using real satellite imagery —
 protecting 4 million people who depend on Krishna River
 for drinking water."

---

## Contact During Event
- Person 1: [phone number]
- Person 2: [phone number]
- Person 3: [phone number]
