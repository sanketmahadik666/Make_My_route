/* ══════════════════════════════════════════════════════════
   ANTIGRAVITY AI — Frontend Application
   Map · API Integration · DOM Manipulation
   ══════════════════════════════════════════════════════════ */

const API_BASE = window.location.origin;

// ─── State ───
let map;
let originMarker = null;
let destMarker = null;
let routePolyline = null;
let stationMarkers = [];
let clickMode = "origin"; // "origin" | "destination"
let originCoords = null;
let destCoords = null;

// ─── Initialize ───
document.addEventListener("DOMContentLoaded", () => {
  initMap();
  initEVProfileListeners();
  initButtonListeners();
  initChipListeners();
  updateUsableEnergyPreview();
  updateBatteryGauge(0.75);
  checkServerHealth();

  // Mobile panel toggle
  const panel = document.getElementById("left-panel");
  panel.addEventListener("click", (e) => {
    if (window.innerWidth < 768 && !panel.classList.contains("expanded")) {
      panel.classList.add("expanded");
    }
  });
});

// ══════════════════════════════════════════════════
// MAP
// ══════════════════════════════════════════════════

function initMap() {
  // Nashik center
  map = L.map("map", {
    center: [20.0063, 73.7900],
    zoom: 13,
    zoomControl: false,
  });

  // Dark tile layer
  L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
    attribution: '&copy; OSM &copy; CARTO',
    subdomains: "abcd",
    maxZoom: 19,
  }).addTo(map);

  // Zoom control — bottom right
  L.control.zoom({ position: "bottomright" }).addTo(map);

  // Click handler
  map.on("click", onMapClick);
}

function onMapClick(e) {
  const { lat, lng: lon } = e.latlng;

  if (clickMode === "origin") {
    setOrigin(lat, lon);
    clickMode = "destination";
  } else {
    setDestination(lat, lon);
    clickMode = "origin";
  }
}

function setOrigin(lat, lon) {
  originCoords = { lat, lon };
  document.getElementById("origin-input").value = `${lat.toFixed(5)}, ${lon.toFixed(5)}`;
  document.getElementById("origin-input").dataset.filled = "true";

  if (originMarker) map.removeLayer(originMarker);
  originMarker = L.circleMarker([lat, lon], {
    radius: 8,
    color: "#00E5A0",
    fillColor: "#0D0F11",
    fillOpacity: 1,
    weight: 3,
  }).addTo(map).bindPopup("Origin");

  updateCalculateButton();
}

function setDestination(lat, lon) {
  destCoords = { lat, lon };
  document.getElementById("dest-input").value = `${lat.toFixed(5)}, ${lon.toFixed(5)}`;
  document.getElementById("dest-input").dataset.filled = "true";

  if (destMarker) map.removeLayer(destMarker);
  destMarker = L.circleMarker([lat, lon], {
    radius: 8,
    color: "#00E5A0",
    fillColor: "#00E5A0",
    fillOpacity: 1,
    weight: 3,
  }).addTo(map).bindPopup("Destination");

  updateCalculateButton();
}

function updateCalculateButton() {
  const btn = document.getElementById("btn-calculate");
  btn.disabled = !(originCoords && destCoords);
}

// ══════════════════════════════════════════════════
// EV PROFILE
// ══════════════════════════════════════════════════

function initEVProfileListeners() {
  // SOC slider
  const socSlider = document.getElementById("f-soc");
  socSlider.addEventListener("input", function () {
    document.getElementById("soc-display").textContent = `${this.value}%`;
    document.getElementById("soc-fill").style.width = `${this.value}%`;
    this.setAttribute("aria-valuenow", this.value);

    const soc = parseFloat(this.value) / 100;
    const fill = document.getElementById("soc-fill");
    fill.setAttribute("data-level", soc > 0.6 ? "full" : soc > 0.2 ? "medium" : "critical");

    updateUsableEnergyPreview();
    updateBatteryGauge(soc);
  });

  // SOH slider
  const sohSlider = document.getElementById("f-soh");
  sohSlider.addEventListener("input", function () {
    document.getElementById("soh-display").textContent = `${this.value}%`;
    document.getElementById("soh-fill").style.width = `${this.value}%`;
    this.setAttribute("aria-valuenow", this.value);
    updateUsableEnergyPreview();
  });

  // Capacity input
  document.getElementById("f-capacity").addEventListener("input", updateUsableEnergyPreview);
}

function updateUsableEnergyPreview() {
  const capacity = parseFloat(document.getElementById("f-capacity").value) || 40;
  const soc = parseFloat(document.getElementById("f-soc").value) / 100;
  const soh = parseFloat(document.getElementById("f-soh").value) / 100;
  const reserve = 0.10;

  const usable = Math.max(capacity * soh * (soc - reserve), 0).toFixed(1);
  document.getElementById("usable-kwh").textContent = `${usable} kWh`;
}

function updateBatteryGauge(soc) {
  const fill = document.getElementById("gauge-fill");
  const ARC_LENGTH = 251.2;
  const offset = ARC_LENGTH * (1 - soc);
  const color = soc > 0.6 ? "var(--soc-full)" : soc > 0.2 ? "var(--soc-medium)" : "var(--soc-low)";

  fill.style.strokeDasharray = ARC_LENGTH;
  fill.style.strokeDashoffset = offset;
  fill.style.stroke = color;

  document.getElementById("gauge-percent").textContent = `${(soc * 100).toFixed(0)}%`;
}

// ══════════════════════════════════════════════════
// CONNECTOR CHIPS
// ══════════════════════════════════════════════════

function initChipListeners() {
  document.getElementById("connector-chips").addEventListener("click", (e) => {
    const chip = e.target.closest(".chip");
    if (!chip) return;
    chip.classList.toggle("chip--selected");
  });
}

function getSelectedConnectors() {
  const chips = document.querySelectorAll("#connector-chips .chip--selected");
  return Array.from(chips).map((c) => c.dataset.connector);
}

// ══════════════════════════════════════════════════
// ROUTE CALCULATION
// ══════════════════════════════════════════════════

function initButtonListeners() {
  document.getElementById("btn-calculate").addEventListener("click", calculateRoute);
}

async function calculateRoute() {
  if (!originCoords || !destCoords) return;

  const btn = document.getElementById("btn-calculate");
  btn.disabled = true;
  btn.textContent = "COMPUTING…";
  btn.dataset.loading = "true";

  // Clear previous results
  clearRouteDisplay();

  const body = {
    origin: originCoords,
    destination: destCoords,
    ev_profile: {
      battery_capacity_kwh: parseFloat(document.getElementById("f-capacity").value) || 40,
      soc_current: parseFloat(document.getElementById("f-soc").value) / 100,
      soh: parseFloat(document.getElementById("f-soh").value) / 100,
      soc_min_reserve: 0.10,
      max_charge_rate_kw: parseFloat(document.getElementById("f-charge-rate").value) || 50,
      connector_types: getSelectedConnectors(),
    },
  };

  try {
    const response = await fetch(`${API_BASE}/api/route`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await response.json();
    onRouteResponse(data);
  } catch (err) {
    console.error("Route API error:", err);
    showFeasibilityBanner("infeasible", { deficitKwh: 0 });
    document.getElementById("banner-subtitle").textContent = `API Error: ${err.message}`;
  } finally {
    btn.disabled = false;
    btn.textContent = "CALCULATE ROUTE";
    btn.dataset.loading = "false";
  }
}

// ══════════════════════════════════════════════════
// RESPONSE HANDLER
// ══════════════════════════════════════════════════

function onRouteResponse(response) {
  // 1. Route card
  if (response.route) {
    populateRouteCard(response);
  }

  // 2. Feasibility banner
  if (!response.feasible && !response.charging_needed) {
    showFeasibilityBanner("infeasible", {
      deficitKwh: response.deficit_kwh,
    });
  } else if (response.charging_needed) {
    showFeasibilityBanner("charging", {
      stopCount: response.charging_stops?.length || 0,
    });
  } else {
    showFeasibilityBanner("feasible");
  }

  // 3. SOC trace
  if (response.soc_trace && response.soc_trace.length > 0) {
    buildSocTrace(response.soc_trace, response.charging_stops || []);
  }

  // 4. Map route polyline
  if (response.route?.geometry) {
    renderRoutePolyline(response.route.geometry);
  }

  // 5. Charging stop cards
  if (response.charging_stops && response.charging_stops.length > 0) {
    renderChargingStopCards(response.charging_stops);
    renderChargingStopMarkers(response.charging_stops);
  }
}

// ══════════════════════════════════════════════════
// ROUTE RESULT CARD
// ══════════════════════════════════════════════════

function populateRouteCard(response) {
  const card = document.getElementById("route-result-card");
  card.dataset.state = response.feasible ? "success" : "infeasible";

  document.getElementById("stat-distance").textContent = response.route.total_distance_km;
  document.getElementById("stat-energy").textContent = response.route.total_energy_kwh;
  document.getElementById("stat-time").textContent = response.route.estimated_time_min;

  const arrivalSoc = response.arrival_soc || 0;
  const el = document.getElementById("stat-arrival-soc");
  el.textContent = `${(arrivalSoc * 100).toFixed(0)}`;
  el.dataset.level = getSocState(arrivalSoc);

  document.getElementById("result-time").textContent = new Date().toLocaleTimeString();
}

// ══════════════════════════════════════════════════
// FEASIBILITY BANNER
// ══════════════════════════════════════════════════

const BANNER_CONFIG = {
  feasible: { title: "Route Ready", icon: "✓", subtitle: "Battery sufficient. No charging stops needed." },
  charging: { title: "Charging Required", icon: "⚡", subtitle: "{n} stop{s} added along your route." },
  infeasible: { title: "Route Infeasible", icon: "✗", subtitle: "Insufficient charge. {deficit} kWh short." },
};

function showFeasibilityBanner(state, data = {}) {
  const banner = document.getElementById("feasibility-banner");
  const config = BANNER_CONFIG[state];
  let subtitle = config.subtitle
    .replace("{n}", data.stopCount || 0)
    .replace("{s}", data.stopCount === 1 ? "" : "s")
    .replace("{deficit}", data.deficitKwh?.toFixed(1) || "0");

  document.getElementById("banner-icon").textContent = config.icon;
  document.getElementById("banner-title").textContent = config.title;
  document.getElementById("banner-subtitle").textContent = subtitle;
  banner.dataset.state = state;

  if (state === "feasible") {
    setTimeout(() => { banner.dataset.state = "hidden"; }, 5000);
  }
}

function dismissBanner() {
  document.getElementById("feasibility-banner").dataset.state = "hidden";
}

// ══════════════════════════════════════════════════
// SOC TRACE BAR
// ══════════════════════════════════════════════════

function buildSocTrace(socTrace, chargingStops) {
  const traceBar = document.getElementById("soc-trace-bar");
  traceBar.dataset.state = "visible";

  const container = document.getElementById("trace-segments");
  const markerLayer = document.getElementById("trace-charger-markers");
  container.innerHTML = "";
  markerLayer.innerHTML = "";

  const chargerNodes = new Set((chargingStops || []).map((s) => s.node));

  // Sample for performance (max 200 segments)
  let displayTrace = socTrace;
  if (socTrace.length > 200) {
    const step = Math.ceil(socTrace.length / 200);
    displayTrace = socTrace.filter((_, i) => i % step === 0 || i === socTrace.length - 1);
  }

  displayTrace.forEach((point, i) => {
    const seg = document.createElement("div");
    seg.className = "trace-segment";
    seg.setAttribute("role", "listitem");
    seg.setAttribute("data-soc-state", getSocState(point.soc));
    seg.setAttribute("tabindex", "0");
    seg.setAttribute("aria-label",
      `SOC ${(point.soc * 100).toFixed(0)}%, ${point.cumulative_kwh?.toFixed(2) || 0} kWh used`);

    seg.addEventListener("mouseenter", () => showTraceTooltip(seg, point));
    seg.addEventListener("mouseleave", hideTraceTooltip);
    container.appendChild(seg);

    if (chargerNodes.has(point.node)) {
      const marker = document.createElement("div");
      marker.className = "trace-charger-marker";
      marker.style.left = `${(i / displayTrace.length) * 100}%`;
      markerLayer.appendChild(marker);
    }
  });

  document.getElementById("trace-start-soc").textContent =
    `${(socTrace[0].soc * 100).toFixed(0)}%`;
  document.getElementById("trace-end-soc").textContent =
    `${(socTrace[socTrace.length - 1].soc * 100).toFixed(0)}%`;
}

function showTraceTooltip(seg, point) {
  const tooltip = document.getElementById("trace-tooltip");
  tooltip.hidden = false;
  tooltip.textContent = `SOC: ${(point.soc * 100).toFixed(1)}% · ${point.cumulative_kwh?.toFixed(2) || 0} kWh`;

  const rect = seg.getBoundingClientRect();
  const wrapper = seg.closest(".trace-segments-wrapper").getBoundingClientRect();
  tooltip.style.left = `${rect.left - wrapper.left + rect.width / 2}px`;
}

function hideTraceTooltip() {
  document.getElementById("trace-tooltip").hidden = true;
}

function getSocState(soc) {
  return soc > 0.6 ? "full" : soc > 0.2 ? "medium" : soc > 0.1 ? "low" : "reserve";
}

// ══════════════════════════════════════════════════
// MAP: ROUTE POLYLINE
// ══════════════════════════════════════════════════

function renderRoutePolyline(geometry) {
  if (routePolyline) map.removeLayer(routePolyline);

  // geometry is [[lon, lat], ...] — Leaflet uses [lat, lon]
  const latLngs = geometry.map(([lon, lat]) => [lat, lon]);

  routePolyline = L.polyline(latLngs, {
    color: "#00E5A0",
    weight: 4,
    opacity: 0.8,
    smoothFactor: 1,
  }).addTo(map);

  map.fitBounds(routePolyline.getBounds(), { padding: [60, 60] });
}

// ══════════════════════════════════════════════════
// CHARGING STOP CARDS
// ══════════════════════════════════════════════════

function renderChargingStopCards(stops) {
  const container = document.getElementById("charging-stops-container");
  container.innerHTML = "";

  stops.forEach((stop, i) => {
    const arrSocPct = (stop.arrival_soc * 100).toFixed(0);
    const depSocPct = (stop.departure_soc * 100).toFixed(0);
    const deltaSoc = (depSocPct - arrSocPct).toFixed(0);

    const card = document.createElement("div");
    card.className = "charging-stop-card";
    card.setAttribute("data-station-id", stop.ocm_id || "");
    card.innerHTML = `
      <div class="stop-card__header">
        <div class="stop-number">${String(i + 1).padStart(2, "0")}</div>
        <div class="stop-info">
          <span class="stop-name">${stop.station_name || "Charging Station"}</span>
          <span class="stop-connector">${stop.connector_type || "—"} · ${stop.power_kw || "—"}kW · ${stop.current_type || ""}</span>
        </div>
        <div class="stop-badge" data-fast="${stop.is_fast_charge}">
          ${stop.is_fast_charge ? "DC FAST" : "AC"}
        </div>
      </div>
      <div style="font-family: var(--font-display); font-size: var(--fs-micro); color: var(--text-tertiary); margin-top: var(--space-2); padding-left: 38px;">
        ${stop.operator ? `⚙ ${stop.operator}` : ""} ${stop.usage_type ? `· ${stop.usage_type}` : ""}
      </div>
      <div class="stop-card__soc-flow">
        <div class="soc-pill soc-pill--arrival">
          <span class="soc-pill__label">ARRIVE</span>
          <span class="soc-pill__value">${arrSocPct}%</span>
        </div>
        <div class="soc-flow-arrow">
          <div class="arrow-line"></div>
          <span class="arrow-charge-label">+${deltaSoc}%</span>
        </div>
        <div class="soc-pill soc-pill--departure">
          <span class="soc-pill__label">DEPART</span>
          <span class="soc-pill__value">${depSocPct}%</span>
        </div>
      </div>
      <div class="stop-card__footer">
        <span class="stop-time-label">CHARGE TIME</span>
        <span class="stop-time-value">${stop.charge_time_min} min</span>
      </div>
    `;
    container.appendChild(card);
  });
}

// ══════════════════════════════════════════════════
// MAP: STATION MARKERS
// ══════════════════════════════════════════════════

function renderChargingStopMarkers(stops) {
  stationMarkers.forEach((m) => map.removeLayer(m));
  stationMarkers = [];

  stops.forEach((stop, i) => {
    if (!stop.lat || !stop.lon) return;

    const marker = L.marker([stop.lat, stop.lon], {
      icon: L.divIcon({
        className: "",
        html: `<div style="
          background: #F5A623;
          color: #0D0F11;
          width: 28px; height: 28px;
          border-radius: 50%;
          display: flex; align-items: center; justify-content: center;
          font-size: 14px; font-weight: bold;
          border: 2px solid #0D0F11;
          box-shadow: 0 0 12px rgba(245,166,35,0.5);
        ">⚡</div>`,
        iconSize: [28, 28],
        iconAnchor: [14, 14],
      }),
    })
      .addTo(map)
      .bindPopup(
        `<b>${stop.station_name || "Charger"}</b><br>` +
        `${stop.power_kw}kW · ${stop.connector_type || "—"} · ${stop.current_type || ""}<br>` +
        `${stop.operator ? `Operator: ${stop.operator}<br>` : ""}` +
        `Arrive: ${(stop.arrival_soc * 100).toFixed(0)}% → Depart: ${(stop.departure_soc * 100).toFixed(0)}%<br>` +
        `Charge: ${stop.charge_time_min} min<br>` +
        `<small style="color:#888">Data: OpenChargeMap (CC BY 4.0)</small>`
      );

    stationMarkers.push(marker);
  });
}

// ══════════════════════════════════════════════════
// HEALTH CHECK
// ══════════════════════════════════════════════════

async function checkServerHealth() {
  try {
    const response = await fetch(`${API_BASE}/api/health`);
    const data = await response.json();
    const dot = document.getElementById("status-dot");
    const text = document.getElementById("system-status-text");

    if (data.status === "ready") {
      dot.dataset.status = "ready";
      text.textContent = `Ready · ${data.graph?.nodes || 0} nodes`;
    } else {
      dot.dataset.status = "degraded";
      text.textContent = data.status || "Degraded";
    }
  } catch {
    document.getElementById("status-dot").dataset.status = "error";
    document.getElementById("system-status-text").textContent = "Server offline";
  }
}

// Periodic health check
setInterval(checkServerHealth, 30_000);

// ══════════════════════════════════════════════════
// CLEAR / RESET
// ══════════════════════════════════════════════════

function clearRouteDisplay() {
  if (routePolyline) { map.removeLayer(routePolyline); routePolyline = null; }
  stationMarkers.forEach((m) => map.removeLayer(m));
  stationMarkers = [];

  document.getElementById("route-result-card").dataset.state = "hidden";
  document.getElementById("soc-trace-bar").dataset.state = "hidden";
  document.getElementById("feasibility-banner").dataset.state = "hidden";
  document.getElementById("charging-stops-container").innerHTML = "";
}

// Event delegation for charging stop cards → pan map to station
document.getElementById("route-result-section").addEventListener("click", (e) => {
  const card = e.target.closest(".charging-stop-card");
  if (!card) return;
  const stationId = card.dataset.stationId;
  const marker = stationMarkers.find((m) => m._popup?._content?.includes(stationId));
  if (marker) {
    map.flyTo(marker.getLatLng(), 15, { duration: 0.5 });
    marker.openPopup();
  }
});
