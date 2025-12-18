# Landeros Clone - Pixel-Perfect AI Agency Website

A complete, pixel-perfect clone of the [Landeros Framer template](https://landeros.framer.website/) built with Next.js 14, TypeScript, and Tailwind CSS.

## 🎯 Project Overview

This is an exact replica of the Landeros AI Agency template featuring:
- **100% Visual Fidelity**: Exact colors, fonts, spacing, and layouts
- **Full Functionality**: Interactive pricing toggle, FAQ accordion, smooth scroll animations
- **Modern Stack**: Next.js 14 App Router, TypeScript, Tailwind CSS
- **Responsive Design**: Fully responsive across all breakpoints
- **Performance Optimized**: Fast loading, smooth animations, optimized fonts

## 🚀 Getting Started

### Prerequisites
- Node.js 18+ installed
- npm or yarn package manager

### Installation

```bash
# Install dependencies
npm install

# Run development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the site.

## 🎨 Design System

### Color Palette
- **Background**: `#0A0A0A` (Deep Black)
- **Card**: `#1F1F1F` (Muted Grey)
- **Accent**: `#D1F86C` (Electric Green)
- **Foreground**: `#FFFFFF` (White)
- **Muted**: `#888888` (Grey)
- **Border**: `#2A2A2A` (Dark Grey)

### Typography
- **Primary Font**: Inter (Body text)
- **Display Font**: Space Grotesk (Headings)

## 📦 Project Structure

```
landeros-clone/
├── app/
│   ├── layout.tsx          # Root layout with fonts
│   ├── page.tsx            # Main page composition
│   └── globals.css         # Global styles & animations
├── components/
│   ├── Navbar.tsx          # Sticky navigation
│   ├── Hero.tsx            # Hero section
│   ├── Process.tsx         # Process steps
│   ├── Projects.tsx        # Project showcase
│   ├── Services.tsx        # AI services with code block
│   ├── TickerTape.tsx      # Animated feature ticker
│   ├── Testimonials.tsx    # Customer reviews
│   ├── Integrations.tsx    # Integration partners
│   ├── Pricing.tsx         # Pricing plans with toggle
│   ├── FAQ.tsx             # Accordion FAQ
│   ├── Strategy.tsx        # Strategy & content section
│   ├── CTA.tsx             # Call-to-action section
│   ├── Community.tsx       # Community links
│   └── Footer.tsx          # Site footer
├── hooks/
│   └── useScrollAnimation.ts  # Scroll reveal hook
└── tailwind.config.ts      # Tailwind configuration
```

## ✨ Features

### Interactive Components
- **Pricing Toggle**: Switch between monthly/yearly with 30% discount calculation
- **FAQ Accordion**: Expandable question/answer sections
- **Scroll Animations**: Fade-in effects on scroll using Intersection Observer
- **Ticker Tape**: Infinite scrolling feature list
- **Card Hover Effects**: Smooth elevation and glow effects

### Technical Highlights
- Server-side rendering with Next.js 14 App Router
- TypeScript for type safety
- Tailwind CSS for utility-first styling
- Custom scroll animations with Intersection Observer API
- Glassmorphism effects with backdrop-filter
- Optimized Google Fonts loading

## 🎯 Content Sections

1. **Hero**: Main headline with CTA
2. **Process**: 3-step workflow with stats
3. **Projects**: Latest project showcase
4. **Services**: AI automation services with code example
5. **Ticker Tape**: Animated feature highlights
6. **Testimonials**: Customer reviews grid
7. **Integrations**: Partner integrations
8. **Pricing**: Three-tier pricing with toggle
9. **FAQ**: Common questions accordion
10. **Strategy**: AI consulting & content creation
11. **CTA**: Job listings and quote
12. **Community**: Discord & Twitter links
13. **Footer**: Site navigation and social links

## 🛠️ Development

```bash
# Run development server
npm run dev

# Build for production
npm run build

# Start production server
npm start

# Run linting
npm run lint
```

## 📝 Notes

- All CTA buttons log actions to console (no backend)
- Images use placeholder gradients (replace with actual assets)
- Smooth scroll behavior enabled globally
- Custom scrollbar styling for dark theme

## 🎨 Customization

To customize colors, edit `tailwind.config.ts`:
```typescript
colors: {
  background: "#0A0A0A",
  accent: "#D1F86C",
  // ... other colors
}
```

To modify animations, edit `app/globals.css`:
```css
.card-hover:hover {
  transform: translateY(-4px);
  box-shadow: 0 20px 40px rgba(209, 248, 108, 0.1);
}
```

## 📄 License

This is a clone project for educational purposes. Original design by Landeros Framer Template.

---

Built with ❤️ using Next.js 14, TypeScript, and Tailwind CSS
