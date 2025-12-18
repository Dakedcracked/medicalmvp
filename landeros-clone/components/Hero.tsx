"use client";

import { useEffect, useState } from "react";
import TypingAnimation from "./TypingAnimation";

export default function Hero() {
  const [scrollY, setScrollY] = useState(0);

  useEffect(() => {
    const handleScroll = () => {
      setScrollY(window.scrollY);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <section className="relative min-h-screen flex items-center justify-center pt-24 pb-20 px-6 overflow-hidden">
      {/* Animated Background glow */}
      <div 
        className="absolute top-1/4 left-1/2 -translate-x-1/2 w-[600px] h-[600px] bg-blue-glow/20 rounded-full blur-[120px] pointer-events-none animate-pulse"
        style={{ transform: `translate(-50%, ${scrollY * 0.5}px)` }}
      />
      <div className="absolute top-1/3 right-1/4 w-[400px] h-[400px] bg-accent-light/10 rounded-full blur-[100px] pointer-events-none animate-pulse" />
      
      <div className="max-w-5xl mx-auto text-center relative z-10">
        <div className="mb-4">
          <span className="inline-block bg-accent/10 border border-accent/30 px-4 py-2 rounded-full text-accent text-sm font-semibold mb-6">
            🧠 Neuron - Next-Gen Medical AI
          </span>
        </div>
        
        <h1 className="font-display text-6xl md:text-7xl lg:text-8xl font-bold mb-6 leading-tight">
          Neuron:{" "}
          <TypingAnimation 
            texts={["Medical Imaging", "AI Diagnosis", "Smart Detection", "Clinical Analysis"]}
            className="text-accent"
          />
        </h1>
        
        <p className="text-lg md:text-xl text-muted max-w-3xl mx-auto mb-10 leading-relaxed animate-fade-in">
          Advanced deep learning algorithms automatically detect and analyze abnormalities 
          in medical images. Get instant, accurate diagnosis assistance with comprehensive 
          deformity detection and clinical insights.
        </p>

        <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-16">
          <a
            href="/analyze"
            className="bg-accent text-background px-8 py-4 rounded-full text-base font-semibold hover:bg-accent/90 transition-all hover:scale-105 shadow-lg shadow-accent/20"
          >
            Try Medical AI Now
          </a>
          <a
            href="#services"
            className="bg-transparent border-2 border-accent text-accent px-8 py-4 rounded-full text-base font-semibold hover:bg-accent/10 transition-all"
          >
            Learn More
          </a>
        </div>

        <div className="flex items-center justify-center gap-8 text-sm text-muted">
          <span>Trusted by big brands around the world</span>
        </div>
      </div>
    </section>
  );
}
