---
name: CreditWise AI
colors:
  surface: '#f8f9fa'
  surface-dim: '#d9dadb'
  surface-bright: '#f8f9fa'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f3f4f5'
  surface-container: '#edeeef'
  surface-container-high: '#e7e8e9'
  surface-container-highest: '#e1e3e4'
  on-surface: '#191c1d'
  on-surface-variant: '#3f484b'
  inverse-surface: '#2e3132'
  inverse-on-surface: '#f0f1f2'
  outline: '#6f797c'
  outline-variant: '#bec8cc'
  surface-tint: '#00687a'
  primary: '#005868'
  on-primary: '#ffffff'
  primary-container: '#0b7285'
  on-primary-container: '#bdf0ff'
  inverse-primary: '#82d2e7'
  secondary: '#006e25'
  on-secondary: '#ffffff'
  secondary-container: '#7ffd8b'
  on-secondary-container: '#007528'
  tertiary: '#a20b1c'
  on-tertiary: '#ffffff'
  tertiary-container: '#c52b31'
  on-tertiary-container: '#ffe1df'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#abedff'
  primary-fixed-dim: '#82d2e7'
  on-primary-fixed: '#001f26'
  on-primary-fixed-variant: '#004e5c'
  secondary-fixed: '#7ffd8b'
  secondary-fixed-dim: '#61df72'
  on-secondary-fixed: '#002106'
  on-secondary-fixed-variant: '#00531a'
  tertiary-fixed: '#ffdad7'
  tertiary-fixed-dim: '#ffb3ae'
  on-tertiary-fixed: '#410005'
  on-tertiary-fixed-variant: '#930016'
  background: '#f8f9fa'
  on-background: '#191c1d'
  surface-variant: '#e1e3e4'
typography:
  display-lg:
    fontFamily: Plus Jakarta Sans
    fontSize: 48px
    fontWeight: '700'
    lineHeight: '1.2'
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Plus Jakarta Sans
    fontSize: 32px
    fontWeight: '600'
    lineHeight: '1.3'
    letterSpacing: -0.01em
  headline-md:
    fontFamily: Plus Jakarta Sans
    fontSize: 24px
    fontWeight: '600'
    lineHeight: '1.4'
  body-lg:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '400'
    lineHeight: '1.6'
  body-md:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: '1.6'
  body-sm:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: '1.5'
  label-bold:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '600'
    lineHeight: '1'
    letterSpacing: 0.05em
  headline-lg-mobile:
    fontFamily: Plus Jakarta Sans
    fontSize: 24px
    fontWeight: '600'
    lineHeight: '1.3'
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  base: 4px
  xs: 8px
  sm: 16px
  md: 24px
  lg: 40px
  xl: 64px
  gutter: 24px
  container-max: 1440px
---

## Brand & Style

The design system is engineered for **CreditWise AI**, a sophisticated Loan Approval Intelligence Dashboard. The brand personality is rooted in "Intelligent Precision"—combining the clinical reliability of financial software with the approachable clarity of a modern SaaS tool. 

The aesthetic follows a **Modern Corporate** style with heavy influences from **Minimalism** and **Tonal Layering**. The goal is to reduce the cognitive load of complex data processing by using vast whitespace, a soft color palette, and a clear visual hierarchy. The emotional response should be one of "Informed Confidence," where loan officers feel empowered by data rather than overwhelmed by it.

## Colors

The palette is anchored by a high-contrast relationship between deep Teal Blue and a clinical white background. 

- **Primary Action**: Use `#0b7285` for all primary CTAs, active states, and brand-critical indicators.
- **Surface Strategy**: The background is `#ffffff`. Use `#f8f9fa` for secondary containers and `#f1f3f5` for subtle structural dividers or tertiary backgrounds.
- **Semantic Feedback**: 
    - **Approval**: Use the Sage Green pair for positive credit scores and final approvals.
    - **Rejection**: Use the Terracotta pair for hard denials or high-risk flags.
    - **Warning**: Use the Amber pair for marginal cases requiring manual review.

## Typography

This design system utilizes a dual-font strategy. **Plus Jakarta Sans** provides a modern, slightly geometric character for headlines, adding a touch of personality to financial data. **Inter** is used for body copy and data labels due to its exceptional legibility at small sizes and high x-height.

- **Line Height**: Maintain a generous `1.6` for body text to ensure high data readability.
- **Letter Spacing**: Use slight negative tracking (-0.01em to -0.02em) for larger headlines to keep them tight and professional. For labels, use `0.05em` letter spacing with uppercase transforms to distinguish them from data values.

## Layout & Spacing

The system uses a **Fluid Grid** with a 12-column structure for desktop. 

- **Desktop (1200px+):** 12 columns, 24px gutters, 40px side margins.
- **Tablet (768px - 1199px):** 8 columns, 16px gutters, 24px side margins.
- **Mobile (Up to 767px):** 4 columns, 16px gutters, 16px side margins.

**The "Airy" Principle**: Elements should rely on large internal padding (`md` or 24px) within cards to prevent the UI from feeling cramped despite high information density. Navigation is a fixed "Sticky" top bar at 72px height to provide a constant point of reference.

## Elevation & Depth

Visual hierarchy is managed through **Low-Contrast Outlines** and **Ambient Shadows**. 

1.  **Level 0 (Base)**: `#ffffff` background.
2.  **Level 1 (Cards/Sections)**: `#ffffff` surface with a 1px border of `#f1f3f5` and a soft shadow: `0 4px 6px rgba(0,0,0,0.03)`.
3.  **Level 2 (Hover/Active)**: On hover, cards should lift slightly using `0 10px 15px rgba(0,0,0,0.04)` and a border transition to `#e9ecef`.
4.  **Level 3 (Modals/Dropdowns)**: Crisp white surfaces with a more pronounced shadow: `0 20px 25px rgba(0,0,0,0.08)`.

Transitions for elevation changes should be consistent: `200ms cubic-bezier(0.4, 0, 0.2, 1)`.

## Shapes

The design system employs a **Rounded** shape language to soften the "sharpness" of financial data and make the AI feel more approachable. 

- **Standard Elements**: 8px (`0.5rem`) for standard inputs, buttons, and small widgets.
- **Large Elements**: 16px (`1rem`) for primary dashboard cards and containers.
- **Interactive Pill**: Use fully rounded (pill-shaped) borders for status badges (Approval/Rejection) and filter chips.

## Components

### Buttons
- **Primary**: Solid `#0b7285`, white text, 8px radius. Subtle scale down (0.98) on click.
- **Secondary**: Ghost style with 1px `#cfd4da` border and `#495057` text.
- **Tertiary**: Transparent background, teal text, no border.

### Cards
Cards are the primary data container. They must include a 24px internal padding and a 1px bottom border for the header section to separate title from content.

### Status Badges
Pill-shaped containers using the semantic color pairs defined in the Colors section. Text should be bolded and 12px for maximum legibility.

### Input Fields
- **Default**: White background, 1px `#dee2e6` border. 
- **Focus**: Border changes to `#0b7285` with a 3px soft teal outer glow (box-shadow).

### Data Widgets
Charts should use a simplified color palette (Teal, Sage, and Gray). Avoid heavy grid lines; use light gray dotted lines for axes to maintain an "airy" feel.

### Lists
Use "Row" style lists for loan applications. Each row should have a subtle hover effect (background change to `#f8f9fa`) and a clear right-aligned action button.