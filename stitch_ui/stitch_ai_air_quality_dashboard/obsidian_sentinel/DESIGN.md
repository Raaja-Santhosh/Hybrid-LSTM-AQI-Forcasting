# Design System Specification: Tactical Obsidian

## 1. Overview & Creative North Star: "The Sentinel Aesthetic"
The Creative North Star for this design system is **The Sentinel Aesthetic**. In the realms of high-precision IoT and Cyber-Security, interface design must transcend mere utility to become a high-fidelity instrument. This system moves away from "web-page" layouts toward "mission-critical dashboards."

We break the "template" look through **Intentional Asymmetry** and **Tonal Depth**. Instead of standard centered grids, use heavy left-leaning alignments with expansive negative space to the right, mimicking a specialized terminal. Overlapping glass containers and high-contrast typography scales create an editorial hierarchy that feels both authoritative and hyper-modern.

## 2. Colors: High-Contrast Tactical Tones
The palette is built on a foundation of deep charcoals and obsidian blacks, punctuated by high-luminance "Active" states and "Critical" alerts.

### The Palette (Material Design Convention)
*   **Background:** `#0c0e14` (Deep Obsidian)
*   **Primary (Active/Optimal):** `#57fe81` (Vibrant Neon Green) — Use for status: GO, secure, and active processes.
*   **Secondary (Pulse/Critical):** `#ff716a` (System Red) — Reserved strictly for hazards, breaches, and critical data points.
*   **Surface Hierarchy:**
    *   `surface_container_lowest`: `#000000`
    *   `surface_container_low`: `#11131a`
    *   `surface_container_high`: `#1d1f27`
    *   `surface_container_highest`: `#23262e`

### Core Layout Rules
*   **The "No-Line" Rule:** 1px solid borders are strictly prohibited for sectioning. Define boundaries solely through background color shifts. A `surface_container_high` card sitting on a `surface` background provides all the definition needed.
*   **Surface Hierarchy & Nesting:** Treat the UI as stacked sheets of frosted obsidian. An outer dashboard module (`surface_container_low`) should house nested data widgets using `surface_container_high` to create organic depth.
*   **The "Glass & Gradient" Rule:** Floating elements must utilize Glassmorphism. Use `surface_variant` at 60% opacity with a `24px` backdrop blur. For primary CTAs, apply a subtle linear gradient from `primary` (#57fe81) to `primary_container` (#00cf59) at a 135° angle to add "soul" to the digital glow.

## 3. Typography: The Manrope Technical Scale
We use **Manrope** for its geometric yet modern grotesque qualities, bridging the gap between "Tech" and "Editorial."

*   **Display (Large/Mid/Small):** 3.5rem / 2.75rem / 2.25rem. Use sparingly for high-level metrics or "Total Threats Neutralized" style hero numbers. Tighten letter-spacing by -2% for a "locked-in" feel.
*   **Headline (Large/Mid/Small):** 2rem / 1.75rem / 1.5rem. Use for section titles. Pair a `headline-sm` with a `label-sm` (all caps) directly above it to create an "Instrument Panel" look.
*   **Body (Large/Mid/Small):** 1rem / 0.875rem / 0.75rem. The workhorse for logs and data descriptions.
*   **Labels:** 0.75rem / 0.6875rem. Essential for technical metadata.

**Editorial Hierarchy:** Contrast is king. Pair a massive `display-lg` metric with a tiny, high-contrast `label-sm` to create the "High-Precision" feel.

## 4. Elevation & Depth: Tonal Stacking
Traditional drop shadows are too "soft" for a security interface. We achieve lift through **Tonal Layering**.

*   **The Layering Principle:** Depth is achieved by "stacking" surface tiers. A floating command palette should be `surface_container_highest`, while the background is `surface_dim`.
*   **Ambient Shadows:** If a floating effect is required (e.g., a critical modal), use a diffused 4% opacity shadow tinted with `primary` (#57fe81) for optimal states, or `secondary` (#ff716a) for critical alerts. This creates a "glow" rather than a shadow.
*   **The "Ghost Border" Fallback:** If a container requires a perimeter for accessibility, use the `outline_variant` token at **15% opacity**. Never use 100% opaque borders.
*   **Glassmorphism:** Apply to any element that "hovers" over data. Use a `backdrop-filter: blur(12px)` to ensure legibility while maintaining the "Obsidian" depth.

## 5. Components

### Buttons & Inputs
*   **Primary Button:** Gradient of `primary` to `primary_container`. Text color: `on_primary`. 1rem (`ROUND_FOUR`) corner radius.
*   **Pulse/Critical Button:** Solid `secondary_container` with `secondary` text. Used only for "Terminate Process" or "Emergency Shutdown."
*   **Input Fields:** Use `surface_container_highest` with no border. On focus, add a "Ghost Border" of `primary` at 40% opacity.

### Data & Feedback
*   **The Pulse Badge:** For critical alerts, use `secondary` (#ff716a) with a subtle outer glow (0px 0px 8px).
*   **Activity Chips:** Small, `ROUND_FOUR` containers using `surface_container_high`. Use a 4px circular dot of `primary` to indicate "Live" status.
*   **Cards & Lists:** **Prohibit divider lines.** Use vertical spacing (24px - 32px) and subtle background shifts between `surface_container_low` and `surface_container_highest` to separate list items.

### Specialized Components
*   **The "Telemetry Feed":** A mono-spaced-style log using `body-sm` on `surface_container_lowest`. Use `primary` for timestamps and `on_surface_variant` for data.
*   **Glow-Indicators:** Linear gauges that use a `primary` to `primary_dim` gradient to show system health.

## 6. Do’s and Don’ts

### Do:
*   **Do** use `secondary` (#ff716a) sparingly. If everything is red, nothing is critical.
*   **Do** use asymmetric layouts. Align headers to the far left and secondary data to the far right.
*   **Do** leverage transparency. Let the deep background bleed through your glass surfaces.
*   **Do** use `ROUND_FOUR` (1rem) consistently for all containers to maintain the "molded" industrial feel.

### Don't:
*   **Don't** use pure white text. Use `on_surface` (#e5e4ed) to reduce eye strain in dark environments.
*   **Don't** use 1px solid borders. They break the "Liquid Obsidian" aesthetic.
*   **Don't** use standard "Success Green." Only use the `primary` vibrant green to denote "Active/Optimal" status.
*   **Don't** use generic icons. Use high-stroke-weight, geometric icons that match the Manrope font's visual weight.