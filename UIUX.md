# UIUX.md — Antigravity AI EV Routing System

## Complete UI/UX Design Specification

> **Design Philosophy:** Precision engineering meets kinetic energy. Every component reflects the system's core truth — this interface _knows_ your battery, your road, and your range. Dark-first, data-dense, zero-ambiguity.

---

## TABLE OF CONTENTS

1. [Design System Foundation](#1-design-system-foundation)
2. [Layout Architecture & DOM Structure](#2-layout-architecture--dom-structure)
3. [Component Library](#3-component-library)
   - 3.1 Map Canvas
   - 3.2 Route Input Panel
   - 3.3 EV Profile Form
   - 3.4 Route Result Card
   - 3.5 SOC Trace Bar
   - 3.6 Charging Stop Card
   - 3.7 Feasibility Status Banner
   - 3.8 Battery Gauge Widget
   - 3.9 Energy Stats Strip
   - 3.10 Station Marker (Map Overlay)
4. [Interaction States](#4-interaction-states)
5. [Motion & Animation System](#5-motion--animation-system)
6. [Responsive Breakpoints](#6-responsive-breakpoints)
7. [Accessibility Requirements](#7-accessibility-requirements)
8. [DOM Manipulation Patterns](#8-dom-manipulation-patterns)
9. [Component Composition Map](#9-component-composition-map)
10. [Design Tokens Reference](#10-design-tokens-reference)

---

## 1. Design System Foundation

### 1.1 Aesthetic Direction

**Theme:** Dark-industrial precision. Think aerospace HUD meets EV dashboard. Not "tech startup landing page" — this is a tool used while driving decisions, not decorating walls.

**Tone:** Confident. Calm. Data-forward. The UI should make the user feel like they have superhuman knowledge of their route before they drive it.

**One unforgettable thing:** The SOC Trace Bar — a glowing, segmented energy timeline that runs horizontally across the bottom of the route card, showing exactly where the battery is at every point on the journey. No other EV app has this.

---

### 1.2 Typography

```
Display / Headers:   "DM Mono"  — monospaced, technical, authoritative
Body / Labels:       "Syne"     — geometric, clean, distinctive (not Inter)
Data / Numbers:      "DM Mono"  — all numeric data uses monospace for alignment
```

**Font Scale (rem-based):**

| Token          | Size     | Weight | Usage                        |
| -------------- | -------- | ------ | ---------------------------- |
| `--fs-display` | 2rem     | 700    | Hero numbers (SOC %, kWh)    |
| `--fs-h1`      | 1.5rem   | 600    | Panel section titles         |
| `--fs-h2`      | 1.125rem | 600    | Card titles, charger names   |
| `--fs-body`    | 0.9rem   | 400    | Body copy, descriptions      |
| `--fs-label`   | 0.75rem  | 500    | Input labels, tags, captions |
| `--fs-micro`   | 0.65rem  | 400    | Metadata, coordinates, IDs   |

**Letter Spacing Rules:**

- All-caps labels: `letter-spacing: 0.08em`
- Monospaced numbers: `letter-spacing: -0.02em` (tighten)
- Body text: `letter-spacing: 0.01em`

---

### 1.3 Color System

```css
:root {
  /* ── Backgrounds — layered z-depth system */
  --bg-base: #080c10; /* deepest — map underlay, modal overlays */
  --bg-surface: #0f1419; /* panels, cards */
  --bg-elevated: #161d26; /* inputs, dropdowns, hover states */
  --bg-overlay: #1e2833; /* tooltips, popovers */

  /* ── Accent — the energy spectrum */
  --accent-primary: #00e5a0; /* electric teal — primary actions, active states */
  --accent-glow: rgba(0, 229, 160, 0.15); /* glow behind accent elements */
  --accent-warning: #f5a623; /* amber — charging stop, low SOC warning */
  --accent-danger: #ff4d4d; /* red — infeasible route, critically low SOC */
  --accent-info: #4a9eff; /* blue — informational, secondary data */

  /* ── Text */
  --text-primary: #e8edf2; /* main readable text */
  --text-secondary: #7a8a99; /* labels, muted data */
  --text-tertiary: #3d4f5c; /* placeholders, disabled */
  --text-inverse: #080c10; /* text on accent backgrounds */

  /* ── Borders */
  --border-subtle: rgba(255, 255, 255, 0.05);
  --border-default: rgba(255, 255, 255, 0.1);
  --border-strong: rgba(255, 255, 255, 0.2);
  --border-accent: rgba(0, 229, 160, 0.4);

  /* ── SOC State Colors (battery level visual language) */
  --soc-full: #00e5a0; /* > 60% */
  --soc-medium: #f5a623; /* 20–60% */
  --soc-low: #ff4d4d; /* < 20% */
  --soc-reserve: #3d4f5c; /* below reserve threshold */
}
```

**Color Usage Rules:**

- `--accent-primary` is ONLY used for: active route polyline, primary CTA button, SOC full state, currently selected element
- Never use more than 2 accent colors in the same component
- Background layers must strictly follow the z-depth order — never use `--bg-base` for a foreground card
- `--accent-danger` triggers only when `soc < soc_min_reserve` OR route is infeasible

---

### 1.4 Spacing System

```css
:root {
  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 16px;
  --space-5: 24px;
  --space-6: 32px;
  --space-7: 48px;
  --space-8: 64px;
}
```

**Spacing Rules for DOM Placement:**

- Panel inner padding: `--space-5` (24px) on all four sides
- Card inner padding: `--space-4` (16px)
- Between sibling cards: `--space-3` (12px) gap
- Input field padding: `--space-3` vertical, `--space-4` horizontal
- Icon-to-label gap: `--space-2` (8px)
- Section dividers: `--space-5` margin above, `--space-4` below

---

### 1.5 Elevation & Shadow System

```css
/* Shadows encode z-depth, not decoration */
--shadow-card: 0 2px 12px rgba(0, 0, 0, 0.4), 0 0 0 1px var(--border-subtle);
--shadow-panel: 0 8px 32px rgba(0, 0, 0, 0.6), 0 0 0 1px var(--border-default);
--shadow-modal: 0 24px 64px rgba(0, 0, 0, 0.8), 0 0 0 1px var(--border-strong);
--shadow-accent:
  0 0 20px rgba(0, 229, 160, 0.2), 0 0 40px rgba(0, 229, 160, 0.08);
--shadow-warning: 0 0 20px rgba(245, 166, 35, 0.2);
--shadow-danger: 0 0 20px rgba(255, 77, 77, 0.2);
```

---

### 1.6 Border Radius

```css
--radius-sm: 4px; /* tags, badges, small chips */
--radius-md: 8px; /* inputs, buttons, small cards */
--radius-lg: 12px; /* main cards, panels */
--radius-xl: 16px; /* modals, large surfaces */
--radius-full: 9999px; /* pills, avatars, circular icons */
```

**Rule:** Never mix radii within the same visual group. All cards in a list share the same radius. Buttons in the same row share the same radius.

---

## 2. Layout Architecture & DOM Structure

### 2.1 Root Layout — Full Screen Split

```
┌─────────────────────────────────────────────────────────────┐
│  [TOPBAR]  logo · region selector · status indicator        │  height: 52px
├──────────────────┬──────────────────────────────────────────┤
│                  │                                          │
│   [LEFT PANEL]   │         [MAP CANVAS]                     │
│   width: 380px   │         flex: 1                          │
│   position:fixed │         position: relative               │
│                  │                                          │
│  ┌────────────┐  │    [ROUTE POLYLINE]                      │
│  │Route Input │  │    [NODE MARKERS]                        │
│  └────────────┘  │    [CHARGER MARKERS]                     │
│  ┌────────────┐  │    [SOC HEATMAP OVERLAY]                 │
│  │EV Profile  │  │                                          │
│  └────────────┘  │    [FLOATING RESULT CARD]                │
│  ┌────────────┐  │    position: absolute                    │
│  │Route Result│  │    bottom: 24px · right: 24px            │
│  └────────────┘  │                                          │
│                  │                                          │
└──────────────────┴──────────────────────────────────────────┘
│  [BOTTOM SOC TRACE BAR]  width: 100%  height: 72px          │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Root DOM Structure

```html
<div id="app" class="app-root">
  <!-- Z-layer 0: Map fills viewport -->
  <div id="map-canvas" class="map-canvas" role="main" aria-label="Route map">
    <div id="map-overlay-soc" class="map-overlay soc-heatmap"></div>
    <div id="map-overlay-route" class="map-overlay route-layer"></div>
    <div id="map-markers" class="map-overlay markers-layer"></div>
  </div>

  <!-- Z-layer 1: Topbar -->
  <header id="topbar" class="topbar" role="banner">
    <div class="topbar__logo">
      <span class="logo-mark">AG</span>
      <span class="logo-name">Antigravity</span>
    </div>
    <div class="topbar__region" id="region-selector"></div>
    <div class="topbar__status" id="system-status"></div>
  </header>

  <!-- Z-layer 2: Left panel -->
  <aside
    id="left-panel"
    class="left-panel"
    role="complementary"
    aria-label="Route planner"
  >
    <section id="route-input-section" class="panel-section"></section>
    <section id="ev-profile-section" class="panel-section"></section>
    <section
      id="route-result-section"
      class="panel-section"
      aria-live="polite"
    ></section>
  </aside>

  <!-- Z-layer 3: Floating map overlays -->
  <div
    id="feasibility-banner"
    class="feasibility-banner"
    role="status"
    aria-live="assertive"
  ></div>
  <div
    id="floating-result"
    class="floating-result-card"
    aria-live="polite"
  ></div>

  <!-- Z-layer 4: SOC Trace pinned to bottom -->
  <div
    id="soc-trace-bar"
    class="soc-trace-bar"
    role="img"
    aria-label="Battery state of charge along route"
  ></div>
</div>
```

### 2.3 CSS Grid & Positioning Rules

```css
.app-root {
  display: grid;
  grid-template-columns: 380px 1fr;
  grid-template-rows: 52px 1fr 72px;
  grid-template-areas:
    "topbar  topbar"
    "panel   map"
    "trace   trace";
  height: 100vh;
  width: 100vw;
  overflow: hidden;
  background: var(--bg-base);
}

.map-canvas {
  grid-area: map;
  position: relative;
  overflow: hidden;
}
.topbar {
  grid-area: topbar;
  position: relative;
  z-index: 10;
}
.left-panel {
  grid-area: panel;
  overflow-y: auto;
  z-index: 5;
}
.soc-trace-bar {
  grid-area: trace;
  z-index: 8;
}

/* Floating card sits on the map */
.floating-result-card {
  position: absolute;
  bottom: calc(72px + 24px); /* above SOC trace bar */
  right: 24px;
  width: 320px;
  z-index: 6;
}
```

**DOM Placement Rules:**

- The left panel uses `overflow-y: auto` — content scrolls inside the panel, NOT the page
- The map canvas is `position: relative` — all map overlays use `position: absolute; inset: 0`
- The SOC Trace Bar uses `position: sticky` within the grid row — it never scrolls
- Floating cards use `position: absolute` relative to `.map-canvas`, not the viewport
- `z-index` layers: map(1) → panel(5) → floating(6) → soc-trace(8) → topbar(10) → modal(100)

---

## 3. Component Library

---

### 3.1 Map Canvas

**Purpose:** The primary data surface. Displays road network, route polyline, SOC-colored segments, charging station markers, origin/destination pins.

#### DOM Attributes

```html
<div
  id="map-canvas"
  class="map-canvas"
  role="main"
  aria-label="Interactive route map"
  data-region="nashik"
  data-zoom="13"
  data-center-lat="20.0059"
  data-center-lon="73.7898"
  data-loading="false"
  data-route-active="false"
></div>
```

| Attribute             | Type   | Values     | Purpose                          |
| --------------------- | ------ | ---------- | -------------------------------- |
| `data-region`         | string | place slug | Identifies active graph region   |
| `data-zoom`           | number | 10–17      | Current map zoom level           |
| `data-center-lat/lon` | float  | WGS84      | Map viewport center              |
| `data-loading`        | bool   | true/false | Shows loading shimmer over map   |
| `data-route-active`   | bool   | true/false | Drives route polyline visibility |

#### Design Attributes

```css
.map-canvas {
  background: #0a1018; /* base before tiles load */
  position: relative;
  overflow: hidden;

  /* Map tile filter: dark mode the basemap */
  filter: brightness(0.75) saturate(0.6) hue-rotate(195deg);
}

/* Route polyline — the main energy path */
.route-polyline {
  stroke: var(--accent-primary);
  stroke-width: 4px;
  stroke-linecap: round;
  stroke-linejoin: round;
  filter: drop-shadow(0 0 6px rgba(0, 229, 160, 0.5));

  /* Animated draw on load */
  stroke-dasharray: 1000;
  stroke-dashoffset: 1000;
  animation: draw-route 1.2s ease-out forwards;
}

/* SOC-colored overlay segments */
.route-segment[data-soc-state="full"] {
  stroke: var(--soc-full);
}
.route-segment[data-soc-state="medium"] {
  stroke: var(--soc-medium);
}
.route-segment[data-soc-state="low"] {
  stroke: var(--soc-low);
}
.route-segment[data-soc-state="reserve"] {
  stroke: var(--soc-reserve);
}
```

#### DOM Manipulation Pattern

```javascript
// Apply SOC color to route segments dynamically
function colorRouteSegments(socTrace, routePolyline) {
  const segments = routePolyline.querySelectorAll(".route-segment");
  segments.forEach((seg, i) => {
    const soc = socTrace[i]?.soc ?? 1.0;
    const state =
      soc > 0.6 ? "full" : soc > 0.2 ? "medium" : soc > 0.1 ? "low" : "reserve";
    seg.setAttribute("data-soc-state", state);
  });
}
```

---

### 3.2 Route Input Panel

**Purpose:** Captures origin and destination coordinates via text input with map-click integration. The primary entry point for a route request.

#### DOM Structure

```html
<section id="route-input-section" class="panel-section">
  <div class="section-header">
    <span class="section-label">ROUTE</span>
  </div>

  <div class="input-group">
    <div class="input-row"
         data-field="origin"
         data-state="empty"         <!-- empty | focused | filled | error -->
         data-input-mode="text">    <!-- text | map-click -->

      <div class="input-row__icon">
        <span class="dot dot--origin" aria-hidden="true"></span>
      </div>
      <div class="input-row__field">
        <label for="origin-input" class="input-label">FROM</label>
        <input
          id="origin-input"
          type="text"
          class="location-input"
          placeholder="Search or click map"
          autocomplete="off"
          aria-label="Origin location"
          aria-describedby="origin-hint"
          data-lat=""
          data-lon=""
          data-resolved="false"
        />
      </div>
      <button class="input-row__map-btn"
              aria-label="Pick origin from map"
              data-target="origin">
        <svg><!-- crosshair icon --></svg>
      </button>
    </div>

    <!-- Connector line between origin and destination -->
    <div class="route-connector" aria-hidden="true">
      <div class="connector-line"></div>
    </div>

    <div class="input-row"
         data-field="destination"
         data-state="empty">
      <div class="input-row__icon">
        <span class="dot dot--destination" aria-hidden="true"></span>
      </div>
      <div class="input-row__field">
        <label for="destination-input" class="input-label">TO</label>
        <input
          id="destination-input"
          type="text"
          class="location-input"
          placeholder="Search or click map"
          autocomplete="off"
          aria-label="Destination location"
          data-lat=""
          data-lon=""
          data-resolved="false"
        />
      </div>
      <button class="input-row__map-btn"
              aria-label="Pick destination from map"
              data-target="destination">
        <svg><!-- crosshair icon --></svg>
      </button>
    </div>
  </div>

  <button id="calculate-route-btn"
          class="cta-button"
          aria-label="Calculate energy-optimal route"
          disabled
          data-loading="false">
    <span class="cta-button__label">Calculate Route</span>
    <span class="cta-button__icon" aria-hidden="true">→</span>
  </button>
</section>
```

#### Design Attributes

```css
.input-row {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  padding: var(--space-3) var(--space-4);
  background: var(--bg-elevated);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  transition:
    border-color 200ms ease,
    box-shadow 200ms ease;
}

.input-row[data-state="focused"] {
  border-color: var(--border-accent);
  box-shadow: 0 0 0 3px var(--accent-glow);
}

.input-row[data-state="filled"] {
  border-color: var(--border-default);
  background: var(--bg-surface);
}

.input-row[data-state="error"] {
  border-color: var(--accent-danger);
  box-shadow: 0 0 0 3px rgba(255, 77, 77, 0.12);
}

/* Origin dot — green */
.dot--origin {
  width: 10px;
  height: 10px;
  border-radius: var(--radius-full);
  background: var(--accent-primary);
  box-shadow: 0 0 8px var(--accent-primary);
}

/* Destination dot — square (different shape to distinguish from origin) */
.dot--destination {
  width: 10px;
  height: 10px;
  border-radius: var(--radius-sm);
  background: var(--accent-warning);
}

/* Vertical connector line between inputs */
.connector-line {
  width: 1px;
  height: 20px;
  margin-left: calc(var(--space-4) + 4px); /* align with dot center */
  background: linear-gradient(
    to bottom,
    var(--accent-primary),
    var(--accent-warning)
  );
  opacity: 0.4;
}

/* CTA Button */
.cta-button {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: var(--space-3) var(--space-4);
  margin-top: var(--space-4);
  background: var(--accent-primary);
  color: var(--text-inverse);
  border: none;
  border-radius: var(--radius-md);
  font-family: "DM Mono", monospace;
  font-size: var(--fs-body);
  font-weight: 600;
  letter-spacing: 0.04em;
  cursor: pointer;
  transition:
    opacity 150ms ease,
    transform 100ms ease,
    box-shadow 200ms ease;
}

.cta-button:hover:not(:disabled) {
  box-shadow: var(--shadow-accent);
  transform: translateY(-1px);
}

.cta-button:active:not(:disabled) {
  transform: translateY(0);
}

.cta-button:disabled {
  opacity: 0.35;
  cursor: not-allowed;
}

.cta-button[data-loading="true"] .cta-button__label::after {
  content: "...";
  animation: ellipsis 1.2s steps(4) infinite;
}
```

#### DOM Manipulation: Enable/Disable CTA

```javascript
// Enable button only when BOTH coordinates are resolved
function syncRouteButtonState() {
  const originResolved =
    document.querySelector('[data-field="origin"] input').dataset.resolved ===
    "true";
  const destResolved =
    document.querySelector('[data-field="destination"] input').dataset
      .resolved === "true";

  const btn = document.getElementById("calculate-route-btn");
  btn.disabled = !(originResolved && destResolved);
}

// When user clicks map to pick location
function setLocationFromMapClick(latlng, target) {
  const input = document.querySelector(`[data-field="${target}"] input`);
  const row = document.querySelector(`[data-field="${target}"]`);

  input.value = `${latlng.lat.toFixed(4)}, ${latlng.lng.toFixed(4)}`;
  input.dataset.lat = latlng.lat;
  input.dataset.lon = latlng.lng;
  input.dataset.resolved = "true";
  row.dataset.state = "filled";

  syncRouteButtonState();
}
```

---

### 3.3 EV Profile Form

**Purpose:** Captures the vehicle's battery parameters. Inputs directly feed the BatteryState model and energy feasibility check.

#### DOM Structure

```html
<section id="ev-profile-section" class="panel-section">
  <div
    class="section-header collapsible"
    data-open="true"
    role="button"
    aria-expanded="true"
    tabindex="0"
  >
    <span class="section-label">EV PROFILE</span>
    <span class="chevron" aria-hidden="true">▾</span>
  </div>

  <div
    class="ev-profile-form"
    id="ev-profile-form"
    role="form"
    aria-label="Electric vehicle parameters"
  >
    <!-- Battery Capacity -->
    <div class="form-field" data-field="battery_capacity_kwh">
      <label class="field-label" for="f-capacity">
        Battery Capacity
        <span class="field-unit">kWh</span>
      </label>
      <input
        id="f-capacity"
        type="number"
        class="field-input"
        min="5"
        max="200"
        step="0.5"
        value="40"
        data-api-key="battery_capacity_kwh"
        aria-describedby="f-capacity-hint"
      />
      <span id="f-capacity-hint" class="field-hint"
        >Rated total battery capacity</span
      >
    </div>

    <!-- SOC — Slider + Number hybrid -->
    <div class="form-field form-field--slider" data-field="soc_current">
      <label class="field-label" for="f-soc">
        Current Charge
        <span class="field-unit soc-live" id="soc-display">75%</span>
      </label>
      <div class="slider-row">
        <div class="slider-track-wrapper">
          <div class="slider-fill" id="soc-fill" style="width: 75%;"></div>
          <input
            id="f-soc"
            type="range"
            class="field-slider"
            min="0"
            max="100"
            step="1"
            value="75"
            data-api-key="soc_current"
            aria-valuemin="0"
            aria-valuemax="100"
            aria-valuenow="75"
            aria-label="State of charge percentage"
          />
        </div>
      </div>
      <div class="slider-markers" aria-hidden="true">
        <span>0%</span><span>25%</span><span>50%</span><span>75%</span
        ><span>100%</span>
      </div>
    </div>

    <!-- SOH -->
    <div class="form-field form-field--slider" data-field="soh">
      <label class="field-label" for="f-soh">
        Battery Health
        <span class="field-unit" id="soh-display">95%</span>
      </label>
      <div class="slider-row">
        <div class="slider-track-wrapper">
          <div
            class="slider-fill soh-fill"
            id="soh-fill"
            style="width: 95%;"
          ></div>
          <input
            id="f-soh"
            type="range"
            class="field-slider"
            min="60"
            max="100"
            step="1"
            value="95"
            data-api-key="soh"
            aria-label="State of health percentage"
          />
        </div>
      </div>
    </div>

    <!-- Max Charge Rate -->
    <div class="form-field" data-field="max_charge_rate_kw">
      <label class="field-label" for="f-charge-rate">
        Max Charge Rate
        <span class="field-unit">kW</span>
      </label>
      <input
        id="f-charge-rate"
        type="number"
        class="field-input"
        min="3.7"
        max="350"
        step="0.1"
        value="50"
        data-api-key="max_charge_rate_kw"
      />
    </div>

    <!-- Connector Types -->
    <div class="form-field" data-field="connector_types">
      <label class="field-label">Connector Types</label>
      <div
        class="connector-chips"
        role="group"
        aria-label="Supported connector types"
      >
        <button
          class="chip chip--selected"
          data-value="CCS2"
          aria-pressed="true"
        >
          CCS2
        </button>
        <button class="chip" data-value="Type2" aria-pressed="false">
          Type 2
        </button>
        <button class="chip" data-value="CHAdeMO" aria-pressed="false">
          CHAdeMO
        </button>
        <button class="chip" data-value="GB/T" aria-pressed="false">
          GB/T
        </button>
      </div>
    </div>

    <!-- Usable Energy Preview (computed, read-only) -->
    <div class="computed-preview" id="usable-energy-preview" aria-live="polite">
      <span class="preview-label">USABLE ENERGY</span>
      <span class="preview-value" id="usable-kwh">24.7 kWh</span>
    </div>
  </div>
</section>
```

#### Design Attributes

```css
/* Slider custom styling */
.slider-track-wrapper {
  position: relative;
  height: 6px;
  background: var(--bg-overlay);
  border-radius: var(--radius-full);
  overflow: visible;
}

.slider-fill {
  position: absolute;
  top: 0;
  left: 0;
  height: 100%;
  border-radius: var(--radius-full);
  background: var(--accent-primary);
  transition: width 80ms linear;
  pointer-events: none;
}

/* SOC fill changes color based on level */
.slider-fill[data-level="low"] {
  background: var(--soc-medium);
}
.slider-fill[data-level="critical"] {
  background: var(--accent-danger);
}

input[type="range"].field-slider {
  position: absolute;
  inset: 0;
  width: 100%;
  appearance: none;
  background: transparent;
  cursor: pointer;
}

input[type="range"]::-webkit-slider-thumb {
  appearance: none;
  width: 16px;
  height: 16px;
  border-radius: var(--radius-full);
  background: var(--accent-primary);
  border: 2px solid var(--bg-surface);
  box-shadow: 0 0 8px rgba(0, 229, 160, 0.5);
  transition: transform 150ms ease;
}

input[type="range"]::-webkit-slider-thumb:hover {
  transform: scale(1.3);
}

/* Connector chips */
.chip {
  padding: var(--space-1) var(--space-3);
  background: var(--bg-elevated);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-sm);
  color: var(--text-secondary);
  font-family: "DM Mono", monospace;
  font-size: var(--fs-label);
  cursor: pointer;
  transition: all 150ms ease;
}

.chip--selected {
  background: var(--accent-glow);
  border-color: var(--border-accent);
  color: var(--accent-primary);
}

/* Computed usable energy preview */
.computed-preview {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: var(--space-3) var(--space-4);
  margin-top: var(--space-4);
  background: var(--bg-elevated);
  border: 1px solid var(--border-accent);
  border-radius: var(--radius-md);
  border-left: 3px solid var(--accent-primary);
}

.preview-label {
  font-family: "DM Mono", monospace;
  font-size: var(--fs-micro);
  letter-spacing: 0.08em;
  color: var(--text-tertiary);
}

.preview-value {
  font-family: "DM Mono", monospace;
  font-size: var(--fs-h2);
  font-weight: 700;
  color: var(--accent-primary);
}
```

#### DOM Manipulation: Live Usable Energy Calculation

```javascript
// Real-time update of usable energy preview
function updateUsableEnergyPreview() {
  const capacity =
    parseFloat(document.getElementById("f-capacity").value) || 40;
  const soc = parseFloat(document.getElementById("f-soc").value) / 100;
  const soh = parseFloat(document.getElementById("f-soh").value) / 100;
  const reserve = 0.1;

  const usable = Math.max(capacity * soh * (soc - reserve), 0).toFixed(1);

  document.getElementById("usable-kwh").textContent = `${usable} kWh`;

  // Update slider fill color based on SOC level
  const fill = document.getElementById("soc-fill");
  const level = soc > 0.6 ? "full" : soc > 0.2 ? "medium" : "critical";
  fill.setAttribute("data-level", level);
}

// Sync slider display value
document.getElementById("f-soc").addEventListener("input", function () {
  document.getElementById("soc-display").textContent = `${this.value}%`;
  document.getElementById("soc-fill").style.width = `${this.value}%`;
  document.getElementById("f-soc").setAttribute("aria-valuenow", this.value);
  updateUsableEnergyPreview();
});
```

---

### 3.4 Route Result Card

**Purpose:** Displays the computed route summary — total distance, energy consumed, travel time, and arrival SOC.

#### DOM Structure

```html
<section
  id="route-result-section"
  class="panel-section"
  aria-live="polite"
  aria-label="Route calculation result"
>
  <div class="route-result-card" data-state="idle">
    <!-- States: idle | loading | success | infeasible | error -->

    <div class="result-header">
      <span class="result-label">ROUTE SUMMARY</span>
      <span class="result-timestamp" id="result-time"></span>
    </div>

    <div class="result-stats-grid">
      <div class="stat-cell">
        <span class="stat-value" id="stat-distance">—</span>
        <span class="stat-unit">km</span>
        <span class="stat-label">DISTANCE</span>
      </div>
      <div class="stat-cell">
        <span class="stat-value" id="stat-energy">—</span>
        <span class="stat-unit">kWh</span>
        <span class="stat-label">ENERGY</span>
      </div>
      <div class="stat-cell">
        <span class="stat-value" id="stat-time">—</span>
        <span class="stat-unit">min</span>
        <span class="stat-label">EST. TIME</span>
      </div>
      <div class="stat-cell stat-cell--arrival">
        <span class="stat-value arrival-soc" id="stat-arrival-soc">—</span>
        <span class="stat-unit">%</span>
        <span class="stat-label">ARRIVAL SOC</span>
      </div>
    </div>
  </div>
</section>
```

#### Design Attributes

```css
.route-result-card {
  background: var(--bg-surface);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-lg);
  padding: var(--space-4);
  box-shadow: var(--shadow-card);
  transition: border-color 300ms ease;
}

.route-result-card[data-state="success"] {
  border-color: var(--border-accent);
  box-shadow:
    var(--shadow-card),
    0 0 0 1px var(--border-accent);
  animation: card-reveal 400ms cubic-bezier(0.22, 1, 0.36, 1) forwards;
}

.route-result-card[data-state="infeasible"] {
  border-color: var(--accent-warning);
  box-shadow: var(--shadow-warning);
}

/* Stats grid: 2×2 on small, 4×1 on wide */
.result-stats-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--space-3);
  margin-top: var(--space-4);
}

.stat-cell {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 2px;
  padding: var(--space-3);
  background: var(--bg-elevated);
  border-radius: var(--radius-md);
}

.stat-value {
  font-family: "DM Mono", monospace;
  font-size: 1.75rem;
  font-weight: 700;
  color: var(--text-primary);
  line-height: 1;
  letter-spacing: -0.03em;
}

.stat-unit {
  font-family: "DM Mono", monospace;
  font-size: var(--fs-micro);
  color: var(--text-tertiary);
}

.stat-label {
  font-family: "Syne", sans-serif;
  font-size: var(--fs-micro);
  letter-spacing: 0.08em;
  color: var(--text-secondary);
  margin-top: var(--space-1);
}

/* Arrival SOC changes color with value */
.arrival-soc[data-level="full"] {
  color: var(--soc-full);
}
.arrival-soc[data-level="medium"] {
  color: var(--soc-medium);
}
.arrival-soc[data-level="low"] {
  color: var(--soc-low);
}
```

---

### 3.5 SOC Trace Bar

**Purpose:** The signature component. A horizontal segmented bar pinned to the screen bottom showing battery SOC at every point along the route. Each segment is color-coded by SOC level. Hovering shows node details.

#### DOM Structure

```html
<div
  id="soc-trace-bar"
  class="soc-trace-bar"
  role="img"
  aria-label="State of charge along route"
>
  <div class="trace-header">
    <span class="trace-label">SOC TRACE</span>
    <span class="trace-range">
      <span id="trace-start-soc">75%</span>
      <span class="trace-arrow">→</span>
      <span id="trace-end-soc">52%</span>
    </span>
  </div>

  <div class="trace-segments-wrapper">
    <div class="trace-segments" id="trace-segments" role="list">
      <!-- Dynamically populated: one .trace-segment per route node -->
      <!-- Example segment: -->
      <!--
      <div class="trace-segment"
           role="listitem"
           data-node="12345"
           data-soc="0.72"
           data-kwh="0.34"
           data-soc-state="full"
           aria-label="Node 12345: SOC 72%, 0.34 kWh consumed"
           tabindex="0">
      </div>
      -->
    </div>

    <!-- Charging stop markers overlaid on trace -->
    <div
      class="trace-charger-markers"
      id="trace-charger-markers"
      aria-hidden="true"
    >
      <!-- Positioned absolutely by JS using segment index -->
    </div>

    <!-- Hover tooltip -->
    <div class="trace-tooltip" id="trace-tooltip" role="tooltip" hidden>
      <span class="tooltip-soc"></span>
      <span class="tooltip-kwh"></span>
      <span class="tooltip-distance"></span>
    </div>
  </div>
</div>
```

#### Design Attributes

```css
.soc-trace-bar {
  background: var(--bg-surface);
  border-top: 1px solid var(--border-default);
  padding: var(--space-2) var(--space-5);
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
}

.trace-segments-wrapper {
  position: relative;
  height: 28px;
}

.trace-segments {
  display: flex;
  height: 100%;
  gap: 1px;
  border-radius: var(--radius-sm);
  overflow: hidden;
}

.trace-segment {
  flex: 1; /* Equal width per node — visually uniform timeline */
  height: 100%;
  border-radius: 0;
  transition:
    filter 150ms ease,
    transform 100ms ease;
  cursor: pointer;
}

.trace-segment[data-soc-state="full"] {
  background: var(--soc-full);
  opacity: 0.85;
}
.trace-segment[data-soc-state="medium"] {
  background: var(--soc-medium);
  opacity: 0.85;
}
.trace-segment[data-soc-state="low"] {
  background: var(--soc-low);
  opacity: 0.85;
}
.trace-segment[data-soc-state="reserve"] {
  background: var(--soc-reserve);
  opacity: 0.6;
}

.trace-segment:hover {
  filter: brightness(1.3);
  transform: scaleY(1.1);
  transform-origin: bottom;
  z-index: 1;
}

/* Charging stop marker on trace */
.trace-charger-marker {
  position: absolute;
  top: -8px;
  width: 2px;
  height: calc(100% + 12px);
  background: var(--accent-warning);
  border-radius: 1px;
}

.trace-charger-marker::before {
  content: "⚡";
  position: absolute;
  top: -16px;
  left: 50%;
  transform: translateX(-50%);
  font-size: 10px;
}
```

#### DOM Manipulation: Build SOC Trace

```javascript
function buildSocTrace(socTrace, chargingStops) {
  const container = document.getElementById("trace-segments");
  const markerLayer = document.getElementById("trace-charger-markers");
  container.innerHTML = "";
  markerLayer.innerHTML = "";

  // Build a Set of charging stop node IDs for quick lookup
  const chargerNodes = new Set(chargingStops.map((s) => s.node));

  socTrace.forEach((point, i) => {
    const seg = document.createElement("div");
    seg.className = "trace-segment";
    seg.setAttribute("role", "listitem");
    seg.setAttribute("data-node", point.node);
    seg.setAttribute("data-soc", point.soc);
    seg.setAttribute("data-soc-state", getSocState(point.soc));
    seg.setAttribute("tabindex", "0");
    seg.setAttribute(
      "aria-label",
      `Node ${point.node}: SOC ${(point.soc * 100).toFixed(0)}%, ${point.cumulative_kwh} kWh used`,
    );

    // Tooltip on hover
    seg.addEventListener("mouseenter", () => showTraceTooltip(seg, point));
    seg.addEventListener("mouseleave", hideTraceTooltip);
    container.appendChild(seg);

    // If this node is a charging stop — overlay marker
    if (chargerNodes.has(point.node)) {
      const marker = document.createElement("div");
      marker.className = "trace-charger-marker";
      // Position as percentage of total width
      marker.style.left = `${(i / socTrace.length) * 100}%`;
      markerLayer.appendChild(marker);
    }
  });

  // Update range labels
  document.getElementById("trace-start-soc").textContent =
    `${(socTrace[0].soc * 100).toFixed(0)}%`;
  document.getElementById("trace-end-soc").textContent =
    `${(socTrace[socTrace.length - 1].soc * 100).toFixed(0)}%`;
}

function getSocState(soc) {
  return soc > 0.6
    ? "full"
    : soc > 0.2
      ? "medium"
      : soc > 0.1
        ? "low"
        : "reserve";
}
```

---

### 3.6 Charging Stop Card

**Purpose:** Displays a single charging stop with station details, arrival/departure SOC, charge time, and power level.

#### DOM Structure

```html
<div
  class="charging-stop-card"
  data-station-id="{{ocm_id}}"
  data-stop-index="1"
  role="article"
  aria-label="Charging stop 1: {{station_name}}"
>
  <div class="stop-card__header">
    <div class="stop-number" aria-hidden="true">01</div>
    <div class="stop-info">
      <span class="stop-name">{{station_name}}</span>
      <span class="stop-connector">{{connector_type}} · {{power_kw}}kW</span>
    </div>
    <div
      class="stop-badge"
      data-fast="{{is_fast_charge}}"
      aria-label="{{is_fast_charge ? 'Fast charge' : 'Standard charge'}}"
    >
      {{is_fast_charge ? 'DC FAST' : 'AC'}}
    </div>
  </div>

  <div class="stop-card__soc-flow">
    <div class="soc-pill soc-pill--arrival" data-soc="{{arrival_soc}}">
      <span class="soc-pill__label">ARRIVE</span>
      <span class="soc-pill__value">{{(arrival_soc*100).toFixed(0)}}%</span>
    </div>
    <div class="soc-flow-arrow" aria-hidden="true">
      <div class="arrow-line"></div>
      <span class="arrow-charge-label">+{{delta_soc}}%</span>
    </div>
    <div class="soc-pill soc-pill--departure">
      <span class="soc-pill__label">DEPART</span>
      <span class="soc-pill__value">{{(departure_soc*100).toFixed(0)}}%</span>
    </div>
  </div>

  <div class="stop-card__footer">
    <span class="stop-time-label">CHARGE TIME</span>
    <span class="stop-time-value">{{charge_time_min}} min</span>
  </div>
</div>
```

#### Design Attributes

```css
.charging-stop-card {
  background: var(--bg-elevated);
  border: 1px solid var(--border-subtle);
  border-left: 3px solid var(--accent-warning);
  border-radius: var(--radius-lg);
  padding: var(--space-4);
  box-shadow: var(--shadow-warning);
}

.stop-number {
  font-family: "DM Mono", monospace;
  font-size: 1.4rem;
  font-weight: 700;
  color: var(--text-tertiary);
  line-height: 1;
}

.stop-badge[data-fast="true"] {
  background: rgba(245, 166, 35, 0.15);
  border: 1px solid var(--accent-warning);
  color: var(--accent-warning);
  font-family: "DM Mono", monospace;
  font-size: var(--fs-micro);
  letter-spacing: 0.08em;
  padding: 2px var(--space-2);
  border-radius: var(--radius-sm);
}

/* SOC transition flow visualization */
.stop-card__soc-flow {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  margin: var(--space-4) 0;
}

.soc-pill {
  flex: 1;
  text-align: center;
  padding: var(--space-2) var(--space-3);
  background: var(--bg-overlay);
  border-radius: var(--radius-md);
}

.soc-pill--arrival .soc-pill__value {
  color: var(--soc-low);
}
.soc-pill--departure .soc-pill__value {
  color: var(--soc-full);
}

.soc-flow-arrow {
  flex: 1;
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.arrow-line {
  width: 100%;
  height: 2px;
  background: linear-gradient(to right, var(--soc-low), var(--soc-full));
  border-radius: 1px;
}

.arrow-charge-label {
  font-family: "DM Mono", monospace;
  font-size: var(--fs-micro);
  color: var(--accent-primary);
  margin-top: 2px;
}
```

---

### 3.7 Feasibility Status Banner

**Purpose:** Full-width contextual banner that appears when a route calculation completes. Green for feasible, amber for feasible-with-charging, red for infeasible.

#### DOM Structure

```html
<div id="feasibility-banner"
     class="feasibility-banner"
     data-state="hidden"    <!-- hidden | feasible | charging | infeasible -->
     role="status"
     aria-live="assertive"
     aria-atomic="true">

  <div class="banner__icon" aria-hidden="true"></div>
  <div class="banner__message">
    <span class="banner__title" id="banner-title"></span>
    <span class="banner__subtitle" id="banner-subtitle"></span>
  </div>
  <button class="banner__dismiss"
          aria-label="Dismiss route status"
          onclick="dismissBanner()">✕</button>
</div>
```

#### Design Attributes

```css
.feasibility-banner {
  position: absolute;
  top: calc(52px + var(--space-4)); /* below topbar */
  left: 50%;
  transform: translateX(-50%) translateY(-20px);
  min-width: 360px;
  max-width: 560px;
  display: flex;
  align-items: center;
  gap: var(--space-3);
  padding: var(--space-3) var(--space-4);
  border-radius: var(--radius-lg);
  opacity: 0;
  pointer-events: none;
  transition:
    transform 300ms cubic-bezier(0.22, 1, 0.36, 1),
    opacity 200ms ease;
  z-index: 50;
}

.feasibility-banner[data-state]:not([data-state="hidden"]) {
  opacity: 1;
  transform: translateX(-50%) translateY(0);
  pointer-events: auto;
}

.feasibility-banner[data-state="feasible"] {
  background: rgba(0, 229, 160, 0.12);
  border: 1px solid rgba(0, 229, 160, 0.35);
  box-shadow: var(--shadow-accent);
}

.feasibility-banner[data-state="charging"] {
  background: rgba(245, 166, 35, 0.12);
  border: 1px solid rgba(245, 166, 35, 0.35);
  box-shadow: var(--shadow-warning);
}

.feasibility-banner[data-state="infeasible"] {
  background: rgba(255, 77, 77, 0.12);
  border: 1px solid rgba(255, 77, 77, 0.35);
  box-shadow: var(--shadow-danger);
}
```

#### DOM Manipulation: Show Banner

```javascript
const BANNER_CONFIG = {
  feasible: {
    title: "Route Ready",
    icon: "✓",
    subtitle: "Battery sufficient. No charging stops needed.",
  },
  charging: {
    title: "Charging Required",
    icon: "⚡",
    subtitle: "{n} stop{s} added along your route.",
  },
  infeasible: {
    title: "Route Infeasible",
    icon: "✗",
    subtitle: "Insufficient charge. {deficit} kWh short.",
  },
};

function showFeasibilityBanner(state, data = {}) {
  const banner = document.getElementById("feasibility-banner");
  const config = BANNER_CONFIG[state];
  let subtitle = config.subtitle
    .replace("{n}", data.stopCount || 0)
    .replace("{s}", data.stopCount === 1 ? "" : "s")
    .replace("{deficit}", data.deficitKwh?.toFixed(1) || 0);

  document.getElementById("banner-title").textContent = config.title;
  document.getElementById("banner-subtitle").textContent = subtitle;
  banner.dataset.state = state;

  // Auto-dismiss success after 5s
  if (state === "feasible") {
    setTimeout(() => {
      banner.dataset.state = "hidden";
    }, 5000);
  }
}
```

---

### 3.8 Battery Gauge Widget

**Purpose:** Circular arc gauge showing current SOC visually. Placed in the EV Profile section as a live preview of battery state.

#### DOM Structure

```html
<div class="battery-gauge" id="battery-gauge" aria-hidden="true">
  <svg class="gauge-svg" viewBox="0 0 120 120" width="120" height="120">
    <!-- Track arc (background) -->
    <path
      class="gauge-track"
      d="M 20 95 A 50 50 0 1 1 100 95"
      fill="none"
      stroke-width="10"
      stroke-linecap="round"
    />
    <!-- Fill arc (dynamic, driven by SOC) -->
    <path
      class="gauge-fill"
      id="gauge-fill"
      d="M 20 95 A 50 50 0 1 1 100 95"
      fill="none"
      stroke-width="10"
      stroke-linecap="round"
      data-soc="0.75"
    />
    <!-- Center labels -->
    <text class="gauge-percent" x="60" y="62" text-anchor="middle">75%</text>
    <text class="gauge-label" x="60" y="78" text-anchor="middle">SOC</text>
  </svg>
</div>
```

#### Design Attributes & Manipulation

```css
.gauge-track {
  stroke: var(--bg-overlay);
}
.gauge-fill {
  stroke: var(--soc-full);
  transition:
    stroke-dashoffset 600ms cubic-bezier(0.4, 0, 0.2, 1),
    stroke 300ms ease;
}
.gauge-percent {
  font-family: "DM Mono", monospace;
  font-size: 20px;
  font-weight: 700;
  fill: var(--text-primary);
}
.gauge-label {
  font-family: "Syne", sans-serif;
  font-size: 10px;
  letter-spacing: 0.08em;
  fill: var(--text-tertiary);
}
```

```javascript
function updateBatteryGauge(soc) {
  const fill = document.getElementById("gauge-fill");
  const ARC_LENGTH = 251.2; // Full arc circumference for this SVG path
  const offset = ARC_LENGTH * (1 - soc);
  const color =
    soc > 0.6
      ? "var(--soc-full)"
      : soc > 0.2
        ? "var(--soc-medium)"
        : "var(--soc-low)";

  fill.style.strokeDasharray = ARC_LENGTH;
  fill.style.strokeDashoffset = offset;
  fill.style.stroke = color;

  fill.previousElementSibling.querySelector(".gauge-percent").textContent =
    `${(soc * 100).toFixed(0)}%`;
}
```

---

### 3.9 Energy Stats Strip

**Purpose:** A compact horizontal strip showing three key computed values: total route energy, energy per km, and estimated cost (if pricing configured).

#### DOM Structure

```html
<div class="energy-stats-strip" role="group" aria-label="Energy statistics">
  <div class="energy-stat">
    <span class="energy-stat__value" id="est-total-kwh">—</span>
    <span class="energy-stat__unit">kWh</span>
    <span class="energy-stat__label">TOTAL ENERGY</span>
  </div>
  <div class="strip-divider" aria-hidden="true"></div>
  <div class="energy-stat">
    <span class="energy-stat__value" id="est-kwh-km">—</span>
    <span class="energy-stat__unit">kWh/km</span>
    <span class="energy-stat__label">EFFICIENCY</span>
  </div>
  <div class="strip-divider" aria-hidden="true"></div>
  <div class="energy-stat">
    <span class="energy-stat__value" id="est-range">—</span>
    <span class="energy-stat__unit">km</span>
    <span class="energy-stat__label">REMAINING RANGE</span>
  </div>
</div>
```

#### Design Attributes

```css
.energy-stats-strip {
  display: flex;
  align-items: center;
  gap: 0;
  background: var(--bg-elevated);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  overflow: hidden;
}

.energy-stat {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: var(--space-3) var(--space-2);
  gap: 2px;
}

.energy-stat__value {
  font-family: "DM Mono", monospace;
  font-size: 1.25rem;
  font-weight: 700;
  color: var(--text-primary);
  letter-spacing: -0.02em;
}

.strip-divider {
  width: 1px;
  height: 36px;
  background: var(--border-subtle);
  flex-shrink: 0;
}
```

---

### 3.10 Station Marker (Map Overlay)

**Purpose:** SVG-based map marker for charging stations, sized by power output and colored by compatibility.

#### DOM Attributes

```html
<div
  class="station-marker"
  data-station-id="{{ocm_id}}"
  data-power="{{power_kw}}"
  data-fast="{{is_fast_charge}}"
  data-compatible="{{compatible}}"
  data-stop-index="{{stop_index_or_null}}"
  role="button"
  tabindex="0"
  aria-label="{{station_name}}: {{power_kw}}kW {{connector_type}} charger"
  style="left: {{x}}px; top: {{y}}px; position: absolute;"
>
  <svg class="marker-icon" viewBox="0 0 24 32">
    <!-- Pin body -->
    <path
      d="M12 0C5.4 0 0 5.4 0 12c0 9 12 20 12 20S24 21 24 12C24 5.4 18.6 0 12 0z"
    />
    <!-- Lightning bolt icon -->
    <text x="12" y="16" text-anchor="middle" font-size="10">⚡</text>
  </svg>
  <span class="marker-power-label">{{power_kw}}kW</span>
</div>
```

#### Design Attributes

```css
.station-marker {
  position: absolute;
  transform: translate(-50%, -100%);
  cursor: pointer;
  z-index: 3;
  transition:
    transform 150ms ease,
    filter 150ms ease;
}

.station-marker:hover {
  transform: translate(-50%, -100%) scale(1.2);
  filter: drop-shadow(0 4px 12px rgba(245, 166, 35, 0.5));
  z-index: 10;
}

/* Default: unselected station */
.marker-icon path {
  fill: var(--bg-surface);
  stroke: var(--accent-warning);
  stroke-width: 1.5;
}

/* Compatible station */
.station-marker[data-compatible="true"] .marker-icon path {
  fill: rgba(245, 166, 35, 0.2);
  stroke: var(--accent-warning);
}

/* Incompatible — muted */
.station-marker[data-compatible="false"] .marker-icon path {
  fill: var(--bg-elevated);
  stroke: var(--text-tertiary);
  opacity: 0.5;
}

/* Assigned as a charging stop in route */
.station-marker[data-stop-index] .marker-icon path {
  fill: var(--accent-warning);
  stroke: var(--accent-warning);
}

.marker-power-label {
  position: absolute;
  top: 100%;
  left: 50%;
  transform: translateX(-50%);
  font-family: "DM Mono", monospace;
  font-size: 9px;
  white-space: nowrap;
  color: var(--text-secondary);
  background: var(--bg-surface);
  padding: 1px 4px;
  border-radius: 2px;
}
```

---

## 4. Interaction States

Every interactive element must implement all 6 states:

| State    | CSS Selector               | Visual Treatment                                                |
| -------- | -------------------------- | --------------------------------------------------------------- |
| Default  | (no modifier)              | Base styling                                                    |
| Hover    | `:hover`                   | Subtle elevation: `translateY(-1px)` + shadow                   |
| Focus    | `:focus-visible`           | `outline: 2px solid var(--accent-primary); outline-offset: 2px` |
| Active   | `:active`                  | `translateY(0)`, scale(0.98)                                    |
| Disabled | `:disabled` / `[disabled]` | `opacity: 0.35; cursor: not-allowed`                            |
| Loading  | `[data-loading="true"]`    | Shimmer animation or spinner on icon                            |

**Loading Shimmer:**

```css
@keyframes shimmer {
  0% {
    background-position: -200% 0;
  }
  100% {
    background-position: 200% 0;
  }
}

.loading-shimmer {
  background: linear-gradient(
    90deg,
    var(--bg-elevated) 25%,
    var(--bg-overlay) 50%,
    var(--bg-elevated) 75%
  );
  background-size: 200% 100%;
  animation: shimmer 1.4s ease infinite;
}
```

---

## 5. Motion & Animation System

```css
/* Core animation timing functions */
--ease-out: cubic-bezier(0.22, 1, 0.36, 1); /* snappy exits */
--ease-in: cubic-bezier(0.64, 0, 0.78, 0); /* deliberate entries */
--ease-spring: cubic-bezier(0.34, 1.56, 0.64, 1); /* bouncy confirmations */

/* Panel slide-in */
@keyframes panel-slide-in {
  from {
    transform: translateX(-20px);
    opacity: 0;
  }
  to {
    transform: translateX(0);
    opacity: 1;
  }
}

/* Card reveal */
@keyframes card-reveal {
  from {
    transform: translateY(8px) scale(0.98);
    opacity: 0;
  }
  to {
    transform: translateY(0) scale(1);
    opacity: 1;
  }
}

/* Route draw */
@keyframes draw-route {
  to {
    stroke-dashoffset: 0;
  }
}

/* SOC trace segments stagger in */
.trace-segment:nth-child(n) {
  animation: segment-appear 300ms var(--ease-out) forwards;
  animation-delay: calc(n * 2ms); /* stagger 2ms per segment */
  opacity: 0;
}

@keyframes segment-appear {
  from {
    transform: scaleY(0);
    opacity: 0;
  }
  to {
    transform: scaleY(1);
    opacity: 1;
  }
}

/* Pulsing accent glow for active route */
@keyframes pulse-glow {
  0%,
  100% {
    box-shadow: var(--shadow-accent);
  }
  50% {
    box-shadow:
      0 0 30px rgba(0, 229, 160, 0.35),
      0 0 60px rgba(0, 229, 160, 0.15);
  }
}
```

**Animation rules:**

- Route polyline draws in over 1.2s — the only animation that should feel "slow"
- SOC trace segments stagger in left-to-right, 2ms per segment
- All interactive feedback (button press, hover) must complete in ≤ 200ms
- Never animate layout properties (`width`, `height`, `margin`) — animate `transform` and `opacity` only
- Respect `prefers-reduced-motion: reduce` — disable all keyframe animations, keep transitions ≤ 100ms

---

## 6. Responsive Breakpoints

| Breakpoint | Width      | Layout Change                                               |
| ---------- | ---------- | ----------------------------------------------------------- |
| Desktop    | ≥ 1024px   | Full split: left panel (380px) + map                        |
| Tablet     | 768–1023px | Collapsible panel: overlay on map, 320px width              |
| Mobile     | < 768px    | Bottom sheet: panel slides up from bottom, map fills screen |

```css
/* Mobile: bottom sheet pattern */
@media (max-width: 767px) {
  .app-root {
    grid-template-columns: 1fr;
    grid-template-rows: 52px 1fr auto;
    grid-template-areas:
      "topbar"
      "map"
      "panel";
  }

  .left-panel {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    height: 50vh;
    transform: translateY(calc(100% - 80px)); /* Peek 80px */
    transition: transform 400ms var(--ease-out);
    border-radius: var(--radius-xl) var(--radius-xl) 0 0;
    z-index: 20;
  }

  .left-panel.expanded {
    transform: translateY(0);
  }
}
```

---

## 7. Accessibility Requirements

- All interactive elements must be reachable by keyboard (`Tab` key order follows visual layout)
- Route result updates use `aria-live="polite"` — screen reader announces without interrupting
- Feasibility banner uses `aria-live="assertive"` — announces immediately (critical info)
- SOC trace bar: `role="img"` with descriptive `aria-label` (cannot be meaningfully navigated segment-by-segment)
- Color is NEVER the sole differentiator — SOC states also use text labels (`FULL`, `LOW`, etc.)
- Minimum contrast ratio: 4.5:1 for body text, 3:1 for large text and UI components (WCAG AA)
- All SVG icons include `aria-hidden="true"` (decorative) or `role="img"` with `aria-label` (informative)
- Focus indicator: `outline: 2px solid var(--accent-primary); outline-offset: 3px` — never `outline: none`

---

## 8. DOM Manipulation Patterns

### 8.1 State Machine Pattern for Components

```javascript
// All components use dataset attributes as state — never class-name booleans
const RouteCard = {
  setState(state) {
    document.querySelector(".route-result-card").dataset.state = state;
    // states: idle | loading | success | infeasible | error
  },

  populate(routeData) {
    this.setState("success");
    document.getElementById("stat-distance").textContent =
      routeData.total_distance_km;
    document.getElementById("stat-energy").textContent =
      routeData.total_energy_kwh;
    document.getElementById("stat-time").textContent =
      routeData.estimated_time_min;

    const arrivalSoc = routeData.arrival_soc;
    const el = document.getElementById("stat-arrival-soc");
    el.textContent = `${(arrivalSoc * 100).toFixed(0)}`;
    el.dataset.level = getSocState(arrivalSoc);
  },
};
```

### 8.2 Event Delegation for Dynamic Content

```javascript
// Use event delegation for dynamically rendered charging stop cards
document
  .getElementById("route-result-section")
  .addEventListener("click", (e) => {
    const stopCard = e.target.closest(".charging-stop-card");
    if (stopCard) {
      const stationId = stopCard.dataset.stationId;
      highlightStationOnMap(stationId);
      panMapToStation(stationId);
    }
  });
```

### 8.3 Coordinated Map + Panel Updates

```javascript
// Single source of truth: when route response arrives, update ALL components
function onRouteResponse(response) {
  // 1. Route card
  RouteCard.populate(response.route);

  // 2. Feasibility banner
  const state = !response.feasible
    ? "infeasible"
    : response.charging_needed
      ? "charging"
      : "feasible";
  showFeasibilityBanner(state, {
    stopCount: response.charging_stops?.length,
    deficitKwh: response.deficit_kwh,
  });

  // 3. SOC trace
  buildSocTrace(response.soc_trace, response.charging_stops || []);

  // 4. Map route
  renderRoutePolyline(response.route.geometry);
  colorRouteSegments(
    response.soc_trace,
    document.querySelector(".route-polyline"),
  );

  // 5. Station markers
  renderChargingStopMarkers(response.charging_stops || []);

  // 6. Charging stop cards in panel
  renderChargingStopCards(response.charging_stops || []);
}
```

---

## 9. Component Composition Map

```
AppRoot
├── Topbar
│   ├── LogoMark
│   ├── RegionSelector
│   └── SystemStatus
├── MapCanvas
│   ├── SocHeatmapOverlay
│   ├── RoutePolyline (SVG)
│   ├── OriginMarker
│   ├── DestinationMarker
│   └── StationMarker[] (dynamic)
├── LeftPanel
│   ├── RouteInputSection
│   │   ├── InputRow (origin)
│   │   ├── RouteConnector
│   │   ├── InputRow (destination)
│   │   └── CalculateButton
│   ├── EVProfileSection
│   │   ├── BatteryGaugeWidget
│   │   ├── FormField (capacity)
│   │   ├── FormField--slider (soc)
│   │   ├── FormField--slider (soh)
│   │   ├── FormField (charge_rate)
│   │   ├── ConnectorChips
│   │   └── UsableEnergyPreview
│   └── RouteResultSection
│       ├── RouteResultCard
│       ├── EnergyStatsStrip
│       └── ChargingStopCard[] (dynamic)
├── FeasibilityBanner
├── SocTraceBar
│   ├── TraceHeader
│   ├── TraceSegments[] (dynamic)
│   └── TraceChargerMarkers[] (dynamic)
└── TraceTooltip (global singleton)
```

---

## 10. Design Tokens Reference

```css
:root {
  /* Typography */
  --font-display: "DM Mono", monospace;
  --font-body: "Syne", sans-serif;
  --fs-display: 2rem;
  --fs-h1: 1.5rem;
  --fs-h2: 1.125rem;
  --fs-body: 0.9rem;
  --fs-label: 0.75rem;
  --fs-micro: 0.65rem;

  /* Spacing */
  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 16px;
  --space-5: 24px;
  --space-6: 32px;
  --space-7: 48px;
  --space-8: 64px;

  /* Radius */
  --radius-sm: 4px;
  --radius-md: 8px;
  --radius-lg: 12px;
  --radius-xl: 16px;
  --radius-full: 9999px;

  /* Transitions */
  --ease-out: cubic-bezier(0.22, 1, 0.36, 1);
  --ease-spring: cubic-bezier(0.34, 1.56, 0.64, 1);
  --dur-fast: 150ms;
  --dur-base: 250ms;
  --dur-slow: 400ms;

  /* Z-index layers */
  --z-map: 1;
  --z-panel: 5;
  --z-float: 6;
  --z-trace: 8;
  --z-topbar: 10;
  --z-banner: 50;
  --z-modal: 100;
}
```

---

_UIUX.md v1.0 — Antigravity AI EV Routing System | Design System Specification_
